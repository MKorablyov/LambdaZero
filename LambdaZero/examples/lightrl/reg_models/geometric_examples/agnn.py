import torch
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv
from argparse import Namespace


class AGNN(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(AGNN, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropout", 0.5)

        self.lin1 = torch.nn.Linear(num_feat, dim)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(dim, dim)

    def forward(self, data):
        x = F.dropout(data.x, training=self.training, p=self.dropout)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin2(x)
        return x


network = ("AGNN", AGNN)