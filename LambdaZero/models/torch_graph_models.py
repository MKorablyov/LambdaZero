import time
from abc import ABC
import os

import numpy as np
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from torch_geometric.nn import GINEConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from LambdaZero.utils import RunningMeanStd

# import torch and torch.nn using ray utils
torch, nn = try_import_torch()

def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor


class GraphMolActorCritic_thv1(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)
        self.max_steps = obs_space.original_space["num_steps"].n

        #print("action space dict", action_space.__dict__)
        #print(model_config)
        #time.sleep(100)

        #self.max_blocks = action_space.max_blocks
        #self.max_branches = action_space.max_branches
        self.num_blocks = model_config['custom_model_config'].get("num_blocks", 125)

        self.rnd_weight = model_config['custom_model_config'].get("rnd_weight", 0)
        self.rnd = (self.rnd_weight != 0)
        rnd_output_dim = model_config['custom_model_config'].get("rnd_output_dim", 1)
        self.rnd_adv_weight = model_config['custom_model_config'].get("rnd_adv_weight", 1.0)
        self.rnd_vf_loss_weight = model_config['custom_model_config'].get("rnd_vf_loss_weight", 1.0)


        self.space = obs_space.original_space['mol_graph']
        self.model = MPNNet_Parametric(self.space.num_node_feat,
                                       kw.get('num_hidden', 64),
                                       self.num_blocks,
                                       self.rnd)
        self._value_out = None
        if self.rnd: 
            self._value_int = None
            self.rnd_target = RND_MPNNet(self.space.num_node_feat,
                                        32,
                                        self.num_blocks,
                                        rnd_output_dim)
            self.rnd_predictor = RND_MPNNet(self.space.num_node_feat,
                                        32,
                                        self.num_blocks,
                                        rnd_output_dim)
            
            self.rnd_obs_stats = RunningMeanStd(shape=(self.space.num_node_feat))
            self.rnd_rew_stats = RunningMeanStd(shape=())
            # Freeze target network
            self.rnd_target.eval()
            for param in self.rnd_target.parameters():
                param.requires_grad = False

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        device = obs["mol_graph"].device

        # Due to the way RLLib forces encodeing observations into
        # fixed length vectors, this part is quite expensive (~timings
        # on big batch), first there's a useless
        # uint8->float->cpu->gpu->cpu->uint8 (~3ms), then we unpack
        # each graph individually which involves a decompress call and
        # more memory copying (~12ms), then we create a Batch from the
        # "data list" of graphs which involves stacking and
        # incrementing arrays (~4ms) and send it back to gpu (<.5ms).
        enc_graphs = obs["mol_graph"].data.cpu().numpy().astype(np.uint8)
        graphs = [self.space.unpack(i) for i in enc_graphs]
        num_steps = obs["num_steps"]
        action_mask = obs["action_mask"]

        data = fast_from_data_list(graphs)
        data = data.to(device)
        # </end of expensive unpacking> The rest of this is the
        # forward pass (~5ms) and then a backward pass + update (which
        # I can't measure directly but seems to take at most ~20ms)

        scalar_outs, data = self.model(data)

        stop_logit = scalar_outs[:, 1:2]
        add_logits = data.stem_preds.reshape((data.num_graphs, -1))
        break_logits = data.jbond_preds.reshape((data.num_graphs, -1))

        actor_logits = torch.cat([stop_logit,
                                  add_logits,
                                  break_logits], 1)

        # mask not available actions
        masked_actions = (1. - action_mask).to(torch.bool)
        actor_logits[masked_actions] = -20  # some very small prob that does not lead to inf

        self._value_out = scalar_outs[:, 0]
        if self.rnd:
            self._value_int = scalar_outs[:, 2]
        return actor_logits, state

    def value_function(self):
        return self._value_out

    def int_value_function(self):
        return self._value_int

    def compute_priors_and_value(self, obs):
        # import pdb; pdb.set_trace();
        obs = torch.tensor([self.preprocessor.transform(obs)]).float()
        obs = obs.to(next(self.parameters()).device)
        input_dict = {
            'obs': restore_original_dimensions(obs, self.obs_space, "torch")
        }
    
        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            return priors, value

    def compute_intrinsic_rewards(self, train_batch):
        obs_ = restore_original_dimensions(train_batch['new_obs'], self.obs_space, "torch")
        device = obs_["mol_graph"].device

        enc_graphs = obs_["mol_graph"].data.cpu().numpy().astype(np.uint8)
        graphs = [self.space.unpack(i) for i in enc_graphs]

        data = fast_from_data_list(graphs)
        data = data.to(device)

        obs_mean = torch.as_tensor(self.rnd_obs_stats.mean, dtype=torch.float32, device=device)
        obs_var = torch.as_tensor(self.rnd_obs_stats.var, dtype=torch.float32, device=device)
        data.x = ((data.x - obs_mean) / (torch.sqrt(obs_var))).clamp(-1, 1)

        target_reward = self.rnd_target(data)
        predictor_reward = self.rnd_predictor(data)
        rnd_loss_ = ((target_reward - predictor_reward) ** 2).sum(1)
        
        rew_mean = torch.as_tensor(self.rnd_rew_stats.mean, dtype=torch.float32, device=rnd_loss_.device)
        rew_var = torch.as_tensor(self.rnd_rew_stats.var, dtype=torch.float32, device=rnd_loss_.device)
        rnd_loss = (rnd_loss_ - rew_mean) / (torch.sqrt(rew_var))
        
        self.rnd_obs_stats.update(data.x.clone().detach().cpu().numpy())
        self.rnd_rew_stats.update(rnd_loss_.clone().detach().cpu().numpy())

        return rnd_loss

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class MPNNet_Parametric(nn.Module):

    def __init__(self, num_feat=14, dim=64, num_out_per_stem=105, rnd=False):
        super().__init__()
        self.lin0 = nn.Linear(num_feat, dim)
        self.num_ops = num_out_per_stem

        net = nn.Sequential(nn.Linear(4, dim), nn.LeakyReLU(), nn.Linear(dim, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.node2stem = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(), nn.Linear(dim, num_out_per_stem))
        self.node2jbond = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(), nn.Linear(dim, 1))

        self.set2set = Set2Set(dim, processing_steps=1)
        if rnd:
            # 3 = [v, simulate logit, v_int]
            self.lin_out = nn.Linear(dim * 2, 3)
        else:
            # 2 = [v, simulate logit]
            self.lin_out = nn.Linear(dim * 2, 2)

    def forward(self, data):
        out = F.leaky_relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        data.stem_preds = self.node2stem(out[data.stem_atmidx])
        data.jbond_preds = self.node2jbond(out[data.jbond_atmidx.flatten()]) \
                               .reshape((data.jbond_atmidx.shape)) \
                               .mean(1) # mean pooling of the 2 jbond atom preds

        out = self.set2set(out, data.batch)
        out = self.lin_out(out)
        return out, data

class RND_MPNNet(nn.Module):
    def __init__(self, num_feat=14, dim=64, num_out_per_stem=105, output_dim=16):
        super().__init__()
        self.lin0 = nn.Linear(num_feat, dim)
        self.num_ops = num_out_per_stem

        net = nn.Sequential(nn.Linear(4, dim), nn.LeakyReLU(), nn.Linear(dim, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=1)
        self.lin_out = nn.Linear(dim * 2, output_dim)

    def forward(self, data):
        out = F.leaky_relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = self.lin_out(out)
        return out


# This is mostly copied from torch_geometric, but with a few checks
# and generalities removed, as well as some speed improvements
def fast_from_data_list(data_list,
                        inckeys=set(['stem_atmidx','edge_index','jbond_atmidx'])):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly.
    """

    keys = data_list[0].keys

    batch = Batch()
    batch.__data_class__ = data_list[0].__class__
    batch.__slices__ = {key: [0] for key in keys}

    for key in keys:
        batch[key] = []

    cumsum = [0] * len(keys)
    batch.batch = []
    for j, key in enumerate(keys):
        cat_dim = 1 if key == 'edge_index' else 0
        slc = batch.__slices__[key]
        bk = batch[key]
        for i, data in enumerate(data_list):
            item = data[key]
            if cumsum[j] > 0:
                item = item + cumsum[j]
            slc.append(item.shape[cat_dim] + slc[-1])
            if key in inckeys:
                cumsum[j] += data.x.shape[0]
            bk.append(item)
            if j == 0:
                batch.batch.append(torch.full((data.x.shape[0], ), i, dtype=torch.long))

    for key in batch.keys:
        item = batch[key][0]
        cd = data_list[0].__cat_dim__(key, item)
        batch[key] = torch.cat(batch[key],
                               dim=cd)

    return batch.contiguous()
