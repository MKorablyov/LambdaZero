"""
Code for an atom-based graph representation and architecture

"""

import warnings

warnings.filterwarnings('ignore')
import os
import torch_geometric.nn as gnn
import LambdaZero.utils
from LambdaZero.examples.gflow.gflow_models.gflow_model_base import GFlowModelBase


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

warnings.filterwarnings('ignore')


from torch.nn import Linear, LayerNorm, ReLU

from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn import MessagePassing, Set2Set, global_add_pool


from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F
import torch


class ActorWrapper(torch.nn.Module):
    is_actor = True

    def __init__(self, cfg: Namespace, model_base):
        super().__init__()

        num_out_per_stem = getattr(cfg, "out_per_stem", 105)

        self._model = model_base
        dim = self._model.per_atom_out_size

        # --> Change 3 Add new ln
        self.node2stem = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, num_out_per_stem))
        self.node2jbond = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, 1))

    def forward(self, data, **kwargs):

        per_mol_out, per_atom_out = self._model(data)

        stem_batch_idx = (
                torch.tensor(data.__slices__['x'], device=per_atom_out.device)[data.stems_batch]
                + data.stems)

        stem_preds = self.node2stem(per_atom_out[stem_batch_idx])

        if hasattr(data, "bonds"):
            bonds_atoms = data.bonds_batch.unsqueeze(1).expand(-1, data.bonds.size(1)).flatten()
            bond_batch_idx = (
                    torch.tensor(data.__slices__['x'], device=per_atom_out.device)[bonds_atoms]
                    + data.bonds.flatten())

            bonds_preds = self.node2jbond(per_atom_out[bond_batch_idx])

            # # mean pooling of the 2 jbond atom preds ( could be just 1)
            bonds_preds = bonds_preds.reshape((data.bonds.shape)).mean(1).unsqueeze(1)
        else:
            bonds_preds = None

        return stem_preds, per_mol_out, bonds_preds


class RegBase(nn.Module):
    def __init__(self, cfg: Namespace, feature_extractor: nn.Module):
        super(RegBase, self).__init__()
        self._feature_extractor = feature_extractor
        dim = self._feature_extractor.dim
        self.num_out = num_out = getattr(cfg, "out_per_mol", 1)

        self._per_atom_out_size = dim

        self.linh1 = nn.Linear(dim, dim)
        self.linh2 = nn.Linear(dim, dim)
        self.linh3 = nn.Linear(dim, dim)
        self.linh4 = nn.Linear(dim, num_out)

        if self.num_out == 1:
            pass
        elif self.num_out == 2:
            self.linh21 = nn.Linear(dim, dim)
            self.linh22 = nn.Linear(dim, dim)
            self.linh23 = nn.Linear(dim, dim)
            self.linh24 = nn.Linear(dim, num_out)
        else:
            raise NotImplemented
    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size

    def forward(self, data, r_steps=None, **kwargs):

        feat = per_atom_out = self._feature_extractor(data)

        # ==========================================================================================

        _act = nn.SiLU
        # rcond = data.rcond.float()[data.batch].unsqueeze(1)
        #
        # in_head = torch.cat([feat, rcond], dim=1)

        out = _act()(self.linh1(feat))
        out = self.linh2(out)
        out = global_add_pool(out, data.batch)
        out = _act()(self.linh3(out))
        out = self.linh4(out)

        if self.num_out == 2:
            out2 = _act()(self.linh21(feat))
            out2 = self.linh22(out2)
            out2 = global_add_pool(out2, data.batch)
            out2 = _act()(self.linh23(out2))
            out2 = self.linh24(out2)
            out = torch.cat([out, out2], dim=-1)
        # ==========================================================================================

        # if self.num_out == 1:
        #     out = out.view(-1)

        return out, per_atom_out


class DeeperGCN(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(DeeperGCN, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = hidden_channels = getattr(cfg, "dim", 128)  # default should be 64
        num_layers = getattr(cfg, "layers", 28)
        self.dropout = getattr(cfg, "dropout", 0.0)  # default is 0.1
        edge_attr_dim = getattr(cfg, "edge_attr_dim", 4)

        self.node_encoder = Linear(num_feat, hidden_channels)
        self.edge_encoder = Linear(edge_attr_dim, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=self.dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, dim)

    def forward(self, data, *args, **kwargs):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)

        return x


class DeeperGCNGflow(GFlowModelBase):
    def __init__(self, cfg: Namespace, **kwargs):
        super(DeeperGCNGflow, self).__init__(cfg, **kwargs)

        repr_type = getattr(cfg, "repr_type", "atom_graph")
        assert repr_type == "atom_graph", "GraphAgent works on atom_graph input"

        self.training_steps = 0

        self._model = ActorWrapper(cfg, RegBase(cfg, DeeperGCN(cfg)))

    def out_to_policy(self, s, stem_o, mol_o):
        stem_e = torch.exp(stem_o)
        mol_e = torch.exp(mol_o[:, 0])
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, state, act, g, stem_o, mol_o):
        stem_e = torch.exp(stem_o)
        mol_e = torch.exp(mol_o[:, 0])
        Z = gnn.global_add_pool(stem_e, state.stems_batch).sum(1) + mol_e
        mol_lsm = torch.log(mol_e / Z)
        stem_lsm = torch.log(stem_e / Z[state.stems_batch, None])
        stem_slices = torch.tensor(state.__slices__['stems'][:-1], dtype=torch.long, device=stem_lsm.device)
        return -(
                stem_lsm[stem_slices + act[:, 1]][
                    torch.arange(act.shape[0]), act[:, 0]] * (act[:, 0] >= 0)
                + mol_lsm * (act[:, 0] == -1))

    def action_negloglikelihood_bonds(self, state, act, bonds_o):
        # Action should be (atom in bond, bond idx out of available bonds)
        bonds_e = torch.exp(bonds_o)
        Z = gnn.global_add_pool(bonds_e, state.bonds_batch).sum(1)
        bonds_lsm = torch.log(bonds_e / Z[state.bonds_batch, None])
        bond_slices = torch.tensor(
            state.__slices__['bonds'], dtype=torch.long, device=bonds_lsm.device
        )
        # Molecules might have 0 bonds -> so 0
        zero_bonds = ((bond_slices[1:] - bond_slices[:-1]) != 0).float()
        bond_slices = (bond_slices[:-1] * zero_bonds).long()

        negll = -(bonds_lsm[bond_slices + act[:, 1]][torch.arange(act.shape[0]), act[:, 0]]) * zero_bonds
        return negll

    def index_output_by_action(self, s, stem_o, mol_o, a):
        # import pdb; pdb.set_trace()
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
        return (
            stem_o[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_o * (a[:, 0] == -1))
    #(stem_o[stem_slices + a[:, 1]][torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0) + mol_o * (a[:, 0] == -1))

    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o

    def forward(self, graph, **kwargs):
        per_stem_out, per_mol_out, per_jbond_out = self._model(graph)
        return per_stem_out, per_mol_out[:, :1]

    def run_model(self, graph, **kwargs):
        per_stem_out, per_mol_out, per_jbond_out = self._model(graph)
        return per_stem_out, per_mol_out, per_jbond_out


    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
