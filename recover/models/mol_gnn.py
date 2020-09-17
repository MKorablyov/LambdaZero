import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
#from LambdaZero.models.torch_models import MPNNetDrop
from recover.models.relation_aware_mlp import RelationAwareMLP
import time

def _rebuild_edge_index_and_graphs(edge_index, graphs):
    '''
    Since running all graphs through MPNN would drastically OOM (have observed
    just sending 32 graphs in OOMing, and there's 4200 total in the dataset!),
    we need to get the specific graphs for the batch and reindex the edge_index
    to match that of the new list of graphs.  This method does both things.
    '''
    # torch.unique guarantees sorted return vals
    # Load batch into memory one at a time to save on memory consumption
    drug_idxs = torch.unique(edge_index, sorted=True)
    graph_batch = Batch.from_data_list([graphs[i] for i in drug_idxs]).to(edge_index.device)

    # Re-index edge_index relative to graph idxs
    bins = np.unique(edge_index.cpu().flatten()) + 1
    re_edge_index = torch.from_numpy(np.digitize(edge_index.cpu(), bins)).to(edge_index.device)

    return re_edge_index, graph_batch

class MolGnnPredictor(torch.nn.Module):
    def __init__(self, linear_channels, num_relation_lin_layers,
                 mpnn_out_dim, gcn_dropout_rate,
                 linear_dropout_rate, num_relations,
                 graph_num_feat, aggr):
    def __init__(self, data, config):
        super().__init__()

        self._aggr = config['aggr']
        if self._aggr not in ['concat', 'hadamard']:
            raise AttributeError('aggr must be one of "concat" or "hadamard"')

        mpnn_out_dim = int(config['embed_dim'])
        self.mpnn = MPNNetDrop(drop_last=True, drop_data=False,
                               drop_weights=False, drop_prob=config['gcn_dropout_rate'],
                               num_feat=data.mol_graphs[0].x.shape[1], dim=mpnn_out_dim)

        # Add two since we're concatenating the concentrations
        drug_channels = mpnn_out_dim + 1 # add 1 for ic50
        in_lin_channels = 2 * drug_channels if self._aggr == 'concat' else drug_channels
        config['linear_channels'].insert(0, in_lin_channels)

        linear_dropout = torch.nn.Dropout(config['linear_dropout_rate'])
        self.predictor = RelationAwareMLP(config['linear_channels'], config['num_relations'],
                                          config['num_relation_lin_layers'], linear_dropout,
                                          batch_norm=False)

    def forward(self, data, batch):
        edge_index, relations, concs, _ = batch
        edge_index, x = _rebuild_edge_index_and_graphs(edge_index, data.mol_graphs)

        x = self.mpnn.get_embed(x, do_dropout=self.training)

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

        elif self._aggr == 'hadamard':
            return x_i * x_j

