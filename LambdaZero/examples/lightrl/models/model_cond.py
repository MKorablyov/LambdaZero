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
from LambdaZero.examples.lightrl.models.model_base import ModelBase


warnings.filterwarnings('ignore')


def one_hot(idx, device, max_cnt):
    hot = torch.zeros(len(idx), max_cnt, device=device).scatter_(1, idx.flatten().unsqueeze(1), 1.)
    return hot


class MPNNcond(ModelBase):
    def __init__(self, cfg: Namespace, obs_shape, action_space):
        super().__init__(cfg, obs_shape, action_space)

        num_feat = getattr(cfg, "num_feat", 14)
        num_vec = getattr(cfg, "num_vec", 0)
        dim = getattr(cfg, "dim", 128,)
        num_out_per_mol = getattr(cfg, "num_out", 2)
        num_out_per_stem = getattr(cfg, "num_out_per_stem", 105)
        levels = getattr(cfg, "levels", 6)
        version = getattr(cfg, "version", 'v2')
        self.max_steps = getattr(cfg, "max_steps", 10)
        self.zero_cond = getattr(cfg, "zero_cond", False)

        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_ops = num_out_per_stem
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps = levels
        assert version in ['v1', 'v2']
        self.version = int(version[1:])

        net = nn.Sequential(nn.Linear(4, 128), nn.LeakyReLU(inplace=True), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        h_size = dim + self.max_steps + 1
        self.lin1 = nn.Linear(h_size, dim * 8)

        self.set2set = Set2Set(h_size, processing_steps=3)
        self.lin3 = nn.Linear(h_size * 2, num_out_per_mol)

        # --> Change 3 Add new ln
        self.node2stem = nn.Sequential(nn.Linear(dim * 8, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, num_out_per_stem))
        self.node2jbond = nn.Sequential(nn.Linear(dim * 8, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, 1))

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

        rsteps = one_hot(inputs.r_steps[data.batch], inputs.r_steps.device, self.max_steps)
        rcond = inputs.rcond.float()[data.batch].unsqueeze(1)
        if self.zero_cond:
            rsteps.zero_()
            rcond.zero_()
            print("Nope")
        out = torch.cat([out, rsteps, rcond], dim=1)

        # --> Change 4 Add new No ln2
        per_atom_out = F.leaky_relu(self.lin1(out))

        # --> Change 5 Extract stem_preds and jbon_preds
        data.stem_preds = self.node2stem(per_atom_out[data.stem_atmidx])
        data.jbond_preds = self.node2jbond(per_atom_out[data.jbond_atmidx.flatten()])\
            .reshape((data.jbond_atmidx.shape)).mean(1)  # mean pooling of the 2 jbond atom preds

        out = self.set2set(out, data.batch)
        sout = self.lin3(out)  # per mol scalar outputs

        stop_logit = sout[:, -1:]
        break_logits = data.jbond_preds.reshape((data.num_graphs, -1))
        add_logits = data.stem_preds.reshape((data.num_graphs, -1))

        actor_logits = torch.cat([stop_logit, break_logits, add_logits], 1)

        value = sout[:, :-1]
        return value, actor_logits, rnn_hxs


