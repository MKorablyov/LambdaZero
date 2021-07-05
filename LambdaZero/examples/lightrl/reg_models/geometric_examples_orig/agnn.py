import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import AGNNConv
from argparse import Namespace


class AGNN(torch.nn.Module):
    def __init__(self, cfg: Namespace):
        super(AGNN, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        dim = getattr(cfg, "dim", 16)
        self.num_out = num_out = getattr(cfg, "num_out", 1)

        self.lin1 = torch.nn.Linear(num_feat, dim)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(dim, num_out)

    def forward(self, data):
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        if self.num_out == 1:
            x = x.view(-1)

        return x,


network = ("AGNN", AGNN)