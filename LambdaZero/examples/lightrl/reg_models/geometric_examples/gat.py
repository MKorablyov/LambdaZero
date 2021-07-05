import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from argparse import Namespace


class GAT(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(GAT, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 8)
        self.dropout = getattr(cfg, "dropout", 0.6)

        self.conv1 = GATConv(num_feat, dim, heads=8, dropout=self.dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(dim * 8, dim, heads=1, concat=False, dropout=self.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


network = ("GAT", GAT)
