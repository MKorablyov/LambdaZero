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


warnings.filterwarnings('ignore')


class MPNNet_v2(nn.Module):
    def __init__(self, cfg: Namespace):
        super(MPNNet_v2, self).__init__()

        num_feat = getattr(cfg, "num_feat", 14)
        num_vec = getattr(cfg, "num_vec", 0)
        dim = getattr(cfg, "dim", 128,)
        levels = getattr(cfg, "levels", 6)
        version = getattr(cfg, "version", 'v2')
        self.num_out = out_size = getattr(cfg, "num_out", 1)

        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_conv_steps = num_conv_steps = levels
        assert version in ['v1', 'v2']
        self.version = int(version[1:])

        net = nn.Sequential(nn.Linear(4, 128), nn.LeakyReLU(inplace=True), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.lin1 = nn.Linear(dim, dim * 8)
        # self.lin2 = nn.Linear(dim * 8, num_out_per_stem) # --> Change 2 (no more)
        self._per_atom_out_size = dim

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin3 = nn.Linear(dim * 2, out_size)

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size

    def forward(self, data):
        if self.version == 1:
            raise NotImplemented
        elif self.version == 2:
            out = F.leaky_relu(self.lin0(data.x))

        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # --> Change 4 Add new No ln2
        per_atom_out = out

        out = self.set2set(out, data.batch)
        sout = self.lin3(out)  # per mol scalar outputs

        value = sout
        if self.num_out == 1:
            value = value.view(-1)

        return value, per_atom_out

