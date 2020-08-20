import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from LambdaZero.examples.drug_comb.model.relation_aware_mlp import RelationAwareMLP
from LambdaZero.examples.drug_comb.model.layers.in_memory_gcn import InMemoryGCN
from LambdaZero.examples.drug_comb.model.layers.gcn_with_attention import GCNWithAttention
import time

class GNN(torch.nn.Module):
    def __init__(self, gcn_channels, rank, linear_channels,
                 num_relation_lin_layers, gcn_dropout_rate,
                 linear_dropout_rate, train_edge_index,
                 val_edge_index, num_relations,
                 num_residual_gcn_layers, gnn_lyr_type, aggr):
        super().__init__()

        self._aggr = aggr
        if self._aggr not in ['concat', 'kroenecker']:
            raise AttributeError('aggr must be one of "concat" or "kroenecker"')

        gcn_dropout = torch.nn.Dropout(gcn_dropout_rate)
        self.convs = []
        for i in range(len(gcn_channels) - 1):
            in_channels  = gcn_channels[i]
            out_channels = gcn_channels[i + 1]

            gnn_lyr = None
            if gnn_lyr_type == 'InMemoryGCN':
                gnn_lyr = InMemoryGCN(in_channels, out_channels,
                                      train_edge_index, val_edge_index)
            elif gnn_lyr_type == 'GCNWithAttention':
                # Note that we here cast rank to an int as if we are using
                # hyperopt, hyperopt's quniform distribution returns discrete values
                # as a float.  We cast as the GCNWithAttention module requires
                # rank to be an int.
                gnn_lyr = GCNWithAttention(in_channels, out_channels,
                                           int(rank), train_edge_index,
                                           val_edge_index, gcn_dropout)

            self.convs.append(gnn_lyr)

        #if num_residual_gcn_layers > len(self.convs):
        #    raise AttributeError(
        #        "num_residual_gcn_layers must be at most " +
        #        "the number of gcn layers"
        #    )

        self.gcn_dropout = gcn_dropout if gnn_lyr_type == 'InMemoryGCN' else None
        self.num_residual_gcn_layers = num_residual_gcn_layers

        linear_dropout = torch.nn.Dropout(linear_dropout_rate)

        lin_input_dim = -1
        embed_size = gcn_channels[-1] if len(gcn_channels) > 0 else 1024
        if self._aggr == 'concat':
            lin_input_dim = (2 * embed_size) + 2
        elif self._aggr == 'kroenecker':
            lin_input_dim = embed_size + 1

        linear_channels.insert(0, lin_input_dim)
        self.predictor = RelationAwareMLP(linear_channels, num_relations,
                                          num_relation_lin_layers, linear_dropout,
                                          batch_norm=False)

    def to(self, device):
        super().to(device)

        for i in range(len(self.convs)):
            self.convs[i] = self.convs[i].to(device)

        self.predictor = self.predictor.to(device)

        return self

    def parameters(self):
        for param in super().parameters():
            yield param

        for conv in self.convs:
            for param in conv.parameters():
                yield param

        for param in self.predictor.parameters():
            yield param

    def forward(self, x, edge_index, relations, concs):
        for i, conv in enumerate(self.convs):
            if i >= len(self.convs) - self.num_residual_gcn_layers:
                x = F.relu(conv(x)) + x
            else:
                x = F.relu(conv(x))

            if self.gcn_dropout is not None:
                x = self.gcn_dropout(x)

        row, col = edge_index.t()
        x_i = torch.cat((x[row], concs[:, 0].view(-1, 1)), dim=1)
        x_j = torch.cat((x[col], concs[:, 1].view(-1, 1)), dim=1)
        z = self._aggregate(x_i, x_j)

        return self.predictor(z, relations)

    def _aggregate(self, x_i, x_j):
        if self._aggr == 'concat':
            # We need to feed in the edges backwards half of the time
            # so that the MLP doesn't just learn the ordering of the edges
            # instead of the combinatioin itself. To do this, we
            # create a random mask of shape (x_i.shape[0], x_i.shape[1]).
            # The line with torch.cat expands the mask according to its
            # first values across all columns of the mask.
            mask = torch.rand(x_i.shape[0], device=x_i.device) >= .5
            mask = torch.cat(x_i.shape[1] * [mask[None].t()], dim=1)

            row = (x_i * mask) + (x_j * ~mask)
            col = (x_i * ~mask) + (x_j * mask)

            return torch.cat((row, col), dim=1)

        elif self._aggr == 'kroenecker':
            return x_i * x_j

