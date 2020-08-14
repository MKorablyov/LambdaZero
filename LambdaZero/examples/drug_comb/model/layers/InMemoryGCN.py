import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
import time

class InMemoryGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_index, aggr="add"):
        super().__init__(aggr=aggr)

        self.w = torch.nn.Linear(in_channels, out_channels)
        self.edge_index = add_remaining_self_loops(edge_index)

        # Compute normalization
        row, col = self.edge_index
        num_unique_nodes = torch.unique(self.edge_index)[0].shape[0]
        deg = degree(col, num_unique_nodes, dtype=torch.long)
        deg_inv_sqrt = deg.pow(-0.5)
        self.norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    def forward(self, x):
        x = self.w(x)
        return self.propagate(self.edge_index, x=x, norm=self.norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

