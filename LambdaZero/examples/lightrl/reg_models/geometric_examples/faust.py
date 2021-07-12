import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from argparse import Namespace


class Faust(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 64)
        edge_attr_dim = getattr(cfg, "edge_attr_dim", 4)
        self.dropout = getattr(cfg, "dropout", 0.5)

        super(Faust, self).__init__()
        self.conv1 = SplineConv(num_feat, dim//2, dim=edge_attr_dim, kernel_size=5, aggr='add')
        self.conv2 = SplineConv(dim//2, dim, dim=edge_attr_dim, kernel_size=5, aggr='add')
        self.conv3 = SplineConv(dim, dim, dim=edge_attr_dim, kernel_size=5, aggr='add')
        self.conv4 = SplineConv(dim, dim, dim=edge_attr_dim, kernel_size=5, aggr='add')
        self.conv5 = SplineConv(dim, dim, dim=edge_attr_dim, kernel_size=5, aggr='add')
        self.conv6 = SplineConv(dim, dim, dim=edge_attr_dim, kernel_size=5, aggr='add')
        self.lin1 = torch.nn.Linear(dim, 256)
        self.lin2 = torch.nn.Linear(256, dim)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin2(x)
        return x


network = ("Faust", Faust)


