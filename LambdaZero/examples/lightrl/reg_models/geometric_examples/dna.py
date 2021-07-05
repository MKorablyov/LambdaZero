import torch
import torch.nn.functional as F
from torch_geometric.nn import DNAConv
from argparse import Namespace


class DNA(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self,
                 cfg: Namespace,
                 hidden_channels=128,
                 num_layers=5,
                 heads=8,
                 groups=16):
        super(DNA, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropout", 0.5)
        if self.dropout == 0:
            drop = 0
        else:
            drop = min(0.8, self.dropout + 0.3)

        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(num_feat, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(
                    hidden_channels, heads, groups, dropout=drop, cached=False))
        self.lin2 = torch.nn.Linear(hidden_channels, dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


network = ("DNA", DNA)