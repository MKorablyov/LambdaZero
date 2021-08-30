import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from argparse import Namespace


class ProtTopkPool(torch.nn.Module):
    atom_feature_extractor = False
    per_atom_output = False

    def __init__(self, cfg: Namespace):
        super(ProtTopkPool, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 128)
        self.dropout = getattr(cfg, "dropout", 0.5)
        self.num_out = num_out = getattr(cfg, "num_out", 1)

        self.conv1 = GraphConv(num_feat, dim)
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = GraphConv(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.conv3 = GraphConv(dim, dim)
        self.pool3 = TopKPooling(dim, ratio=0.8)

        self.lin1 = torch.nn.Linear(dim*2, dim)
        self.lin2 = torch.nn.Linear(dim, dim//2)
        self.lin3 = torch.nn.Linear(dim//2, num_out)

        self._per_atom_out_size = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        if self.num_out == 1:
            x = x.view(-1)

        return x, None

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size


network = ("ProtTopkPool", ProtTopkPool)

