import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from LambdaZero.models.torch_models import MPNNetDrop
from LambdaZero.examples.drug_comb.model.relation_aware_mlp import RelationAwareMLP
import time

def _aggregate(x_i, x_j):
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

class MolGnnPredictor(torch.nn.Module):
    def __init__(self, linear_channels, num_relation_lin_layers,
                 mpnn_out_dim, gcn_dropout_rate,
                 linear_dropout_rate, num_relations,
                 graph_num_feat, aggr):
        super().__init__()

        self._aggr = aggr
        if self._aggr not in ['concat', 'hadamard']:
            raise AttributeError('aggr must be one of "concat" or "hadamard"')

        self.mpnn = MPNNetDrop(drop_last=True, drop_data=False,
                               drop_weights=False, drop_prob=gcn_dropout_rate,
                               num_feat=graph_num_feat, dim=mpnn_out_dim)

        # Add two since we're concatenating the concentrations
        drug_channels = mpnn_out_dim + 1 # add 1 for ic50
        in_lin_channels = 2 * drug_channels if self._aggr == 'concat' else drug_channels
        linear_channels.insert(0, in_lin_channels)

        linear_dropout = torch.nn.Dropout(linear_dropout_rate)
        self.predictor = RelationAwareMLP(linear_channels, num_relations,
                                          num_relation_lin_layers, linear_dropout,
                                          batch_norm=False)

    def to(self, device):
        super().to(device)
        self.predictor = self.predictor.to(device)
        return self

    def parameters(self):
        for param in super().parameters():
            yield param

        for param in self.predictor.parameters():
            yield param

    def forward(self, x, edge_index, relations, concs):
        x = self.mpnn.get_embed(x, do_dropout=self.training)

        row, col = edge_index.t()
        x_i = torch.cat((x[row], concs[:, 0].view(-1, 1)), dim=1)
        x_j = torch.cat((x[col], concs[:, 1].view(-1, 1)), dim=1)
        z = _aggregate(x_i, x_j)

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

        elif self._aggr == 'hadamard':
            return x_i * x_j

