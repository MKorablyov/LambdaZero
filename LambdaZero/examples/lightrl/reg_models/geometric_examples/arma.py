from argparse import Namespace

import torch
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv


class Arma(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(Arma, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropout", 0.25)

        self.conv1 = ARMAConv(num_feat, dim, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=self.dropout)

        self.conv2 = ARMAConv(dim, dim, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=self.dropout,
                              act=lambda x: x)

    def forward(self, data):
        drop = min(self.dropout * 2, 0.8)
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training, p=drop)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=drop)
        x = self.conv2(x, edge_index)
        return x


network = ("Arma", Arma)
