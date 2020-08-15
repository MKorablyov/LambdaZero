import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree, add_remaining_self_loops
from LambdaZero.examples.drug_comb.model.layers.RelationAwareLinear import RelationAwareLinear
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
        for i in range(num_layers - num_relation_layers):
            in_channels  = layer_channels[i]
            out_channels = layer_channels[i + 1]

            modules.append(torch.nn.Linear(in_channels, out_channels))

            if i != num_layers - 1:
                modules.append(torch.nn.ReLU())
                modules.append(dropout)
                if batch_norm:
                    modules.append(torch.nn.BatchNorm1d(out_channels))

        self.dropout = dropout
        self.non_relation_pred = torch.nn.Sequential(*modules)
        self.relation_lyrs = [
            RelationAwareLinear(num_relations, layer_channels[j], layer_channels[j + 1])
            for j in range(i, num_layers)
        ]

        self.bn = None
        if batch_norm:
            self.bn = [
                torch.nn.BatchNorm1d(layer_channels[j])
                for k in range(num_layers - num_relation_layers, num_layers)
            ]

    def forward(self, x, relations):
        x = self.non_relation_pred(x)

        num_rel_lyrs = len(self.relation_lyrs)
        for i in range(num_rel_lyrs):
            x = self.relation_lyrs[i](x, relations)

            if i != num_rel_lyrs - 1:
                x = F.relu(x)
                x = self.dropout(x)

                if self.bn is not None:
                    x = self.bn[i](x)

        return x

