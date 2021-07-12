import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import FiLMConv
from argparse import Namespace


class FILM(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace, hidden_channels=320, num_layers=4):
        super(FILM, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)

        self.dropout = getattr(cfg, "dropout", 0.1)

        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(num_feat, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(FiLMConv(hidden_channels, hidden_channels))
        self.convs.append(FiLMConv(hidden_channels, dim, act=None))

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(BatchNorm1d(hidden_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


network = ("FILM", FILM)