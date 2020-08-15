import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from LambdaZero.examples.drug_comb.model.layers.GCNWithAttention import GCNWithAttention
from LambdaZero.examples.drug_comb.model.layers.RelationAwareMLP import RelationAwareMLP
import time

class GNNWithAttention(torch.nn.Module):
    def __init__(self, gcn_channels, rank, linear_channels,
                 num_relation_lin_layers, gcn_dropout_rate,
                 linear_dropout_rate, train_edge_index,
                 val_edge_index, num_relations,
                 num_residual_gcn_layers, aggr):
        super().__init__()

        self._aggr = aggr
        if self._aggr not in ['concat', 'kroenecker']:
            raise AttributeError('aggr must be one of "concat" or "kroenecker"')

        gcn_dropout = torch.nn.Dropout(gcn_dropout_rate)
        self.convs = []
        for i in range(len(gcn_channels) - 1):
            in_channels  = gcn_channels[i]
            out_channels = gcn_channels[i + 1]

            self.convs.append(GCNWithAttention(in_channels, out_channels,
                                               rank, train_edge_index,
                                               val_edge_index, gcn_dropout))

        if num_residual_gcn_layers > len(self.convs):
            raise AttributeError(
                "num_residual_gcn_layers must be at most " +
                "the number of gcn layers"
            )

        linear_dropout = torch.nn.Dropout(linear_dropout_rate)

        lin_input_dim = -1
        if self._aggr == 'concat':
            lin_input_dim = 2 * gcn_channels[-1]
        else if self._aggr == 'kroenecker':
            lin_input_dim = gcn_channels[-1]

        linear_channels.insert(0, lin_input_dim)
        self.predictor = RelationAwareMLP(linear_channels, num_relations
                                          num_relation_lin_layers, linear_dropout,
                                          batch_norm=False)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if i > len(self.convs) - num_residual_gcn_layers:
                x += F.relu(self.conv(x))
            else:
                x = F.relu(self.conv(x))

            x = self.gcn_dropout(x)

        row, col = edge_index
        z = self._aggregate(x[row], x[col])

        return self.predictor(z, relations)

    def _aggregate(self, x_i, x_j):
        if self._aggr == 'concat':
            idx = torch.rand(x_i.shape[0]) >= .5

            row = x_i[idx] + x_j[~idx]
            col = x_i[~idx] + x_j[idx]

            return torch.cat((row, col), dim=1)

        else if self._aggr == 'kroenecker':
            return x_i * x_j

