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
        self.num_blocks = model_config['custom_model_config'].get("num_blocks", 135)

        self.space = obs_space.original_space['mol_graph']
        self.model = MPNNet_Parametric(self.space.num_node_feat,
                                       kw.get('num_hidden', 64),
                                       self.num_blocks)
        self._value_out = None


    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        device = obs["mol_graph"].device

        num_steps = obs["num_steps"]
        action_mask = obs["action_mask"]
        mol_graph = obs["mol_graph"]

        # todo: the if loop below is not very elegant but is to solve a problem of rllib sending dummy batches
        # if dummy batch initialize all with 0s
        if torch.equal(mol_graph, torch.zeros_like(mol_graph)):
            actor_logits = torch.zeros_like(action_mask)
            self._value_out = torch.zeros_like(action_mask[:,0])
        else:
            # Due to the way RLLib forces encoding observations into
            # fixed length vectors, this part is quite expensive (~timings
            # on big batch), first there's a useless
            # uint8->float->cpu->gpu->cpu->uint8 (~3ms), then we unpack
            # each graph individually which involves a decompress call and
            # more memory copying (~12ms), then we create a Batch from the
            # "data list" of graphs which involves stacking and
            # incrementing arrays (~4ms) and send it back to gpu (<.5ms).
            enc_graphs = obs["mol_graph"].data.cpu().numpy().astype(np.uint8)
            graphs = [self.space.unpack(i) for i in enc_graphs]
            #print("graph 0", graphs[0])

            data = fast_from_data_list(graphs)
            data = data.to(device)
            # </end of expensive unpacking> The rest of this is the
            # forward pass (~5ms) and then a backward pass + update (which
            # I can't measure directly but seems to take at most ~20ms)

            scalar_outs, data = self.model(data)
            stop_logit = scalar_outs[:, 1:2]
            break_logits = data.jbond_preds.reshape((data.num_graphs, -1))
            add_logits = data.stem_preds.reshape((data.num_graphs, -1))

            actor_logits = torch.cat([stop_logit,
                                      break_logits,
                                      add_logits], 1)

            #print("break logits", break_logits.shape, "add logits", add_logits.shape,
            #      "actor_logits", actor_logits.shape, "action mask", action_mask.shape)

            # mask not available actions
            masked_actions = (1. - action_mask).to(torch.bool)
            actor_logits[masked_actions] = -20  # some very small prob that does not lead to inf
            self._value_out = scalar_outs[:, 0]
        return actor_logits, state

    def value_function(self):
        return self._value_out

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

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class MPNNet_Parametric(nn.Module):
    def __init__(self, num_feat=16, dim=64, num_out_per_stem=105):
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
        # 2 = [v, simulate logit]
        self.lin_out = nn.Linear(dim * 2, 2)

    def forward(self, data):
        out = F.leaky_relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            self.gru.flatten_parameters()
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        data.jbond_preds = self.node2jbond(out[data.jbond_atmidx.flatten()]) \
                               .reshape((data.jbond_atmidx.shape)) \
                               .mean(1) # mean pooling of the 2 jbond atom preds
        data.stem_preds = self.node2stem(out[data.stem_atmidx])
        #print("jbond preds", data.jbond_preds.shape,
        #      "stem preds", data.stem_preds.shape, "jbond_atmidx", data.jbond_atmidx)
        out = self.set2set(out, data.batch)
        out = self.lin_out(out)
        return out, data



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
