"""
Code for an atom-based graph representation and architecture

"""
import warnings
warnings.filterwarnings('ignore')
import os
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

from LambdaZero import chem
from LambdaZero.chem import atomic_numbers
from lightrl.models.model_base import ModelBase


warnings.filterwarnings('ignore')


class MPNNetMultihead(ModelBase):
    def __init__(self, cfg: Namespace, obs_shape, action_space):
        super().__init__(cfg, obs_shape, action_space)

        num_feat = getattr(cfg, "num_feat", 14)
        num_vec = getattr(cfg, "num_vec", 0)
        dim = getattr(cfg, "dim", 128,)
        num_out_per_mol = getattr(cfg, "num_out_per_mol", 2)
        num_out_per_stem = getattr(cfg, "num_out_per_stem", 105)
        levels = getattr(cfg, "levels", 6)
        version = getattr(cfg, "version", 'v2')

        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_ops = num_out_per_stem
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps = levels
        assert version in ['v1', 'v2']
        self.version = int(version[1:])

        net = nn.Sequential(nn.Linear(4, 128), nn.LeakyReLU(inplace=True), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.lin1 = nn.Linear(dim, dim * 8)
        # self.lin2 = nn.Linear(dim * 8, num_out_per_stem) # --> Change 2 (no more)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin3 = nn.Linear(dim * 2, num_out_per_mol)

        # --> Change 3 Add new ln
        self.node2stem = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 8, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, num_out_per_stem))
            for _ in range(50)])
        self.node2jbond = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 8, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, 1))
            for _ in range(50)])

    def forward(self, inputs, rnn_hxs, masks, vec_data=None):
        data = inputs.mol_graph

        if self.version == 1:
            batch_vec = vec_data[data.batch]
            out = F.leaky_relu(self.lin0(torch.cat([data.x, batch_vec], 1)))
        elif self.version == 2:
            out = F.leaky_relu(self.lin0(data.x))

        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # --> Change 4 Add new No ln2
        per_atom_out = F.leaky_relu(self.lin1(out))

        # import pdb; pdb.set_trace()
        break_logits = torch.zeros(data.num_graphs, 6, device=inputs.seed.device)
        add_logits = torch.zeros(data.num_graphs, 2100, device=inputs.seed.device)
        for im, mol_id in enumerate(inputs.seed.unique()):
            filter = inputs.seed == mol_id

            # --> Change 5 Extract stem_preds and jbon_preds
            stem_preds = self.node2stem[im](per_atom_out[data.stem_atmidx]).reshape((data.num_graphs, -1))
            jbond_preds = self.node2jbond[im](per_atom_out[data.jbond_atmidx.flatten()]).reshape((data.jbond_atmidx.shape)).mean(1).reshape((data.num_graphs, -1))  # mean pooling of the 2 jbond atom preds
            break_logits[filter] += jbond_preds[filter]
            add_logits[filter] += stem_preds[filter]

        out = self.set2set(out, data.batch)
        sout = self.lin3(out)  # per mol scalar outputs

        stop_logit = sout[:, 1:2]
        # break_logits = data.jbond_preds.reshape((data.num_graphs, -1))
        # add_logits = data.stem_preds.reshape((data.num_graphs, -1))

        actor_logits = torch.cat([stop_logit, break_logits, add_logits], 1)

        value = sout[:, :1]
        return value, actor_logits, rnn_hxs
