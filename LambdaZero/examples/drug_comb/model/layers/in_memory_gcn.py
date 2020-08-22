import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
import time

class InMemoryGCN(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 train_edge_index, val_edge_index, aggr="add"):
        super().__init__(aggr=aggr)

        self.w = torch.nn.Linear(in_channels, out_channels)
        self.train_edge_index = add_remaining_self_loops(train_edge_index)[0]
        self.val_edge_index = add_remaining_self_loops(val_edge_index)[0]

        norms = []
        # Compute normalization
        for edge_index in [self.train_edge_index, self.val_edge_index]:
            row, col = edge_index
            num_unique_nodes = torch.unique(edge_index).shape[0]
            deg = degree(col, num_unique_nodes, dtype=torch.long)
            deg_inv_sqrt = deg.pow(-0.5)
            norms.append(deg_inv_sqrt[row] * deg_inv_sqrt[col])

        self.train_norm = norms[0]
        self.val_norm   = norms[1]

    def forward(self, x):
        x = self.w(x)
        edge_index = self.train_edge_index if self.training else self.val_edge_index
        norm = self.train_norm if self.training else self.val_norm
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

