from abc import ABC
import os

import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from torch_geometric.nn import GINEConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

# import torch and torch.nn using ray utils
from analyses_and_plots.profiling import profile

torch, nn = try_import_torch()



def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor




class GraphMolActorCritic_thv1(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        print('model_config', model_config)
        self.max_steps = obs_space.original_space["num_steps"].n
        self.max_blocks = action_space.max_blocks
        self.max_branches = action_space.max_branches
        self.num_blocks = action_space.num_blocks

        self.space = obs_space.original_space['mol_graph']
        self.model = MPNNet_Parametric(self.space.num_node_feat,
                                       model_config.get('num_hidden', 32),
                                       self.num_blocks)

        self._value_out = None

    @profile
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        device = obs["mol_graph"].device

        # This part might be slow, because I'm unzipping a lot of data
        # into graphs, but from some tests it seems to be relatively
        # marginal, at most 10% extra cost
        graphs = [self.space.unpack(i.data.cpu().numpy().astype(np.uint8)) for i in obs["mol_graph"]]
        num_steps = obs["num_steps"]
        action_mask = obs["action_mask"]

        # print out which device various objects are on
        mol_graph_device = obs["mol_graph"].device
        action_mask_device = action_mask.device
        model_device = next(self.model.parameters()).device
        print(f"mol_graph device in GraphMolActorCritic_thv1.forward is {mol_graph_device}")
        print(f"action_mask device in GraphMolActorCritic_thv1.forward is {action_mask_device}")
        print(f"the MPNN model device in GraphMolActorCritic_thv1.forward is {model_device}")

        data = Batch.from_data_list(graphs, ['stem_atmidx', 'jbond_atmidx',
                                             'stem_preds', 'jbond_preds']).to(device)

        scalar_outs, data = self.model(data)


        stop_logit = scalar_outs[:, 1:2]
        # Both these methods of zero-padding the logits are about as
        # fast, I'm not sure where else I could be doing something
        # wrong/better
        if 1:
            add_logits = torch.zeros((data.num_graphs, self.max_branches, self.num_blocks),
                                     device=action_mask.device)
            break_logits = torch.zeros((data.num_graphs, self.max_blocks-1), device=action_mask.device)
            for i, g in enumerate(data.to_data_list()):
                add_logits[i, :len(g.stem_atmidx)] = g.stem_preds
                break_logits[i, :len(g.jbond_atmidx)] = g.jbond_preds

        if 0:
            add_logits = []
            break_logits = []
            for i, g in enumerate(data.to_data_list()):
                al = g.stem_preds
                if al.shape[0] < self.max_branches:
                    al = torch.cat( [al,
                                     torch.zeros((self.max_branches - al.shape[0],
                                                  self.num_blocks), device=device)], 0)
                bl = g.jbond_preds
                if bl.shape[0] < self.max_blocks-1:
                    bl = torch.cat( [bl,
                                     torch.zeros((self.max_blocks-1 - bl.shape[0]),
                                                 device=device)], 0)
                add_logits.append(al)
                break_logits.append(bl)
            add_logits = torch.stack(add_logits)
            break_logits = torch.stack(break_logits)

        actor_logits = torch.cat([stop_logit,
                                  add_logits.reshape((data.num_graphs, -1)),
                                  break_logits], 1)

        # mask not available actions
        masked_actions = (1. - action_mask).to(torch.bool)
        actor_logits[masked_actions] = -20  # some very small prob that does not lead to inf

        self._value_out = scalar_outs[:, 0]
        return actor_logits, state

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        raise NotImplementedError()
        obs = torch.tensor([self.preprocessor.transform(obs)]).float().cuda()
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

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
    def __init__(self, num_feat=14, dim=64, num_out_per_stem=105):
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

    @profile
    def forward(self, data):
        # unravel this code to see where the time is spent
        x = data.x
        intermediate = self.lin0(x)
        out = F.leaky_relu(intermediate)
        h = out.unsqueeze(0)

        for i in range(6):
            # unravel this code to see where the time is spent
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            intermediate = self.conv(out, edge_index, edge_attr)
            m = F.leaky_relu(intermediate)

            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        data.stem_preds = self.node2stem(out[data.stem_atmidx])
        data.jbond_preds = self.node2jbond(out[data.jbond_atmidx.flatten()]) \
                               .reshape((data.jbond_atmidx.shape)) \
                               .mean(1) # mean pooling of the 2 jbond atom preds

        out = self.set2set(out, data.batch)
        out = self.lin_out(out)
        return out, data
