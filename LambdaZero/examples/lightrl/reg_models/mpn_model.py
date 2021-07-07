import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
from argparse import Namespace


class MPNNet(nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def __init__(self, cfg: Namespace, **kwargs):
        super(MPNNet, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        dim = getattr(cfg, "dim", 64)
        self.num_out = out_size = getattr(cfg, "num_out", 1)

        self.lin0 = nn.Linear(num_feat, dim)

        h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, h, aggr='mean')
        self.gru = nn.GRU(dim, dim)
        self._per_atom_out_size = dim

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, out_size)

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size

    def forward(self, data):
        out = nn.functional.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = nn.functional.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        per_atom_out = out
        out = self.set2set(out, data.batch)
        out = nn.functional.relu(self.lin1(out))
        out = self.lin2(out)

        if self.num_out == 1:
            out = out.view(-1)

        return out, per_atom_out
