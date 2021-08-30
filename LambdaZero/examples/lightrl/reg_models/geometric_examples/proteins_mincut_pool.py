from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from argparse import Namespace


class ProtMincutPool(torch.nn.Module):
    atom_feature_extractor = False

    def __init__(self, cfg: Namespace):
        super(ProtMincutPool, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 32)
        self.num_out = num_out = getattr(cfg, "num_out", 1)

        average_nodes = 80  # TODO hardcoded

        self.conv1 = GCNConv(num_feat, dim)
        num_nodes = ceil(0.5 * average_nodes)
        self.pool1 = Linear(dim, num_nodes)

        self.conv2 = DenseGraphConv(dim, dim)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(dim, num_nodes)

        self.conv3 = DenseGraphConv(dim, dim)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, num_out)

        self._per_atom_out_size = dim

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = per_atom_out = F.relu(self.conv1(x, edge_index))

        x, mask = to_dense_batch(x, batch)

        adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)

        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = F.relu(self.conv2(x, adj))

        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        if self.num_out == 1:
            x = x.view(-1)

        return x, per_atom_out

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size


network = ("ProtMincutPool", ProtMincutPool)
