import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_geometric.nn import GENConv, DeepGCNLayer
from argparse import Namespace


class DeeperGCN(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(DeeperGCN, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = hidden_channels = getattr(cfg, "dim", 64)
        num_layers = getattr(cfg, "layers", 28)
        self.dropout = getattr(cfg, "dropout", 0.1)
        edge_attr_dim = getattr(cfg, "edge_attr_dim", 4)

        self.node_encoder = Linear(num_feat, hidden_channels)
        self.edge_encoder = Linear(edge_attr_dim, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=self.dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return x


network = ("DeeperGCN", DeeperGCN)