import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from LambdaZero.examples.drug_comb.model.layers.InMemoryGCN import InMemoryGCN
from LambdaZero.examples.drug_comb.model.layers.LowRankAttention import LowRankAttention
import time

class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rank,
                 train_edge_index, val_edge_index, dropout):
        super().__init__()

        self.conv = InMemoryGCN(in_channels, out_channels, train_edge_index, val_edge_index)
        self.attention = LowRankAttention(in_channels, rank)
        self.dim_reduce = torch.nn.Linear(out_channels + (2 * rank), out_channels)

        self.dropout = dropout

    def forward(self, x):
        x_local = F.relu(self.conv(x))
        x_local = self.dropout(x_local)

        x_global = self.attention(x)

        return self.dim_reduce(torch.cat((x_local, x_global), dim=1))

