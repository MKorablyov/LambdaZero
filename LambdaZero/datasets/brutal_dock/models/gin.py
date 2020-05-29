from torch import nn as nn
from torch_geometric.nn import GINEConv, global_add_pool

from LambdaZero.datasets.brutal_dock.models.model_base import ModelBase


class GraphIsomorphismNet(ModelBase):
    def __init__(self,
                 name: str = "GIN",
                 node_feat: int = 14,
                 edge_feat: int = 4,
                 gin_layers: int = 2,
                 gin_size: int = 128,
                 gin_mlp_hidden: int = 128,
                 gin_hidden: int = 128,
                 linear_hidden: int = 128,
                 out_size: int = 1
                 ):
        """
        message passing neural network.

        Args:
            name (str, optional): name of this model
            node_feat (int, optional): number of input features. Defaults to 14.
            edge_feat (int, optional): number of edge features. Defaults to 3.
            gin_layers (int, optional): number of GIN layers. Defaults to 2.
            gin_size (int, optional): size of GCN embedding size. Defaults to 128.
            gin_mlp_hidden (int, optional): size of hidden layer in GIN MLP. Defaults to 128.
            gin_hidden (int, optional): output size of GIN. Defaults to 128.
            linear_hidden (int, optional): hidden size in fully-connected network. Defaults to 128.
            out_size (int, optional): output size. Defaults to 1.
        """
        super(GraphIsomorphismNet, self).__init__()
        self.name = name

        self.node_lin0 = nn.Linear(node_feat, gin_size)

        self.edge_lin0 = nn.Linear(edge_feat, gin_size)

        # GINConv doesn't allow for edge feature - use GINEConv instead

        gin_mlps = [nn.Sequential(
            nn.Linear(gin_size, gin_mlp_hidden),
            nn.ReLU(),
            nn.Linear(gin_mlp_hidden, gin_hidden))]

        for _ in range(gin_layers - 1):
            gin_mlps.append(nn.Sequential(
                nn.Linear(gin_hidden, gin_mlp_hidden),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden, gin_hidden)
            ))

        self.graphconv = nn.ModuleList([GINEConv(gmlp, train_eps=True) for gmlp in gin_mlps])

        # update edge features for each layer
        edge_mlps = [nn.Sequential(
            nn.Linear(gin_size, gin_mlp_hidden),
            nn.ReLU(),
            nn.Linear(gin_mlp_hidden, gin_hidden))]

        for _ in range(gin_layers - 1):
            edge_mlps.append(nn.Sequential(
                nn.Linear(gin_hidden, gin_mlp_hidden),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden, gin_hidden)))

        self.edge_mlps = nn.ModuleList(edge_mlps)

        self.fully_connected = nn.Sequential(
            nn.Linear(gin_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, out_size)
        )

    def forward(self, data):
        node_out = self.node_lin0(data.x)
        edge_out = self.edge_lin0(data.edge_attr)

        # graph convolution
        for gin, edge_update in zip(self.graphconv, self.edge_mlps):
            node_out = gin(node_out, data.edge_index, edge_out)
            edge_out = edge_update(edge_out)

        # graph pooling - use sum pooling
        out = global_add_pool(node_out, data.batch)

        out = self.fully_connected(out)

        return out
