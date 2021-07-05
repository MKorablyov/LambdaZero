import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
from argparse import Namespace


class TaGCN(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(TaGCN, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropout", 0.5)

        self.conv1 = TAGConv(num_feat, dim)
        self.conv2 = TAGConv(dim, dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x


network = ("TaGCN", TaGCN)