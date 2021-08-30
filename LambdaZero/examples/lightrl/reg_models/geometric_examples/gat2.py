import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from argparse import Namespace
from torch_geometric.nn import GATConv


class GAT2(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(GAT2, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropoutdropout", 0.5)

        self.conv1 = GATConv(num_feat, dim)
        self.conv2 = GATConv(dim, dim)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        edge_weight = None  # GCNConv doesn't work with our edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


network = ("GAT2", GAT2)
