import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
import time

class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rank, edge_index, dropout):
        super().__init__()

        self.conv = InMemoryGCN(in_channels, out_channels, edge_index)
        self.attention = LowRankAttention(in_channels, rank)
        self.dim_reduce = torch.nn.Linear(out_channels + (4 * rank), out_channels)

        self.dropout = dropout

    def forward(self, x):
        x_local = F.relu(self.conv(x))
        x_local = self.dropout(x_local)

        x_global = self.attention(x)

        return self.dim_reduce(torch.cat((x_local, x_global), dim=1))

