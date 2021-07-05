import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from argparse import Namespace


class CoraNet(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(CoraNet, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        edge_attr_dim = getattr(cfg, "edge_attr_dim", 4)
        self.dropout = getattr(cfg, "dropout", 0.5)

        self.conv1 = SplineConv(num_feat, dim, dim=edge_attr_dim, kernel_size=2)
        self.conv2 = SplineConv(dim, dim, dim=edge_attr_dim, kernel_size=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_attr)
        return x


network = ("CoraNet", CoraNet)
