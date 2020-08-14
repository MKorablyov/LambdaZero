import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree, add_remaining_self_loops
import time

class RelationAwareMLP(torch.nn.Module):
    def __init__(self, layer_channels, num_relations,
                 num_relation_layers, dropout, batch_norm = False):
        """
        Arguments
        ---------
        layer_channels : List[int]
        The number of layers in the LinkPredictor will be len(layer_channels) - 1.
        Further, the ith layer of the predictor will have in_channels and out_channels
        equal according to
            in_channels_i  = layer_channels[i]
            out_channels_i = layer_channels[i + 1]
        """
        super().__init__()

        modules = []
        num_layers = len(layer_channels) - 1
        for i in range(num_layers):
            in_channels  = layer_channels[i]
            out_channels = layer_channels[i + 1]

            # Relation layers come at the end of the layers list
            if i < num_layers - num_relation_layers:
                modules.append(torch.nn.Linear(in_channels, out_channels))
            else:
                modules.append(RelationAwareLinear(num_relations, in_channels, out_channels))

            if i != num_layers - 1:
                modules.append(torch.nn.ReLU())
                modules.append(dropout)
                if batch_norm:
                    modules.append(torch.nn.BatchNorm1d(out_channels))

        self.pred = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.pred(x)

