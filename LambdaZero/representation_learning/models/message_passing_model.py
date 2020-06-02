from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, Set2Set

from LambdaZero.representation_learning.models.model_base import ModelBase


class MessagePassingNet(ModelBase):
    def __init__(self,
                 name: str = "MPNN",
                 node_feat: int = 14,
                 edge_feat: int = 4,
                 gcn_size: int = 128,
                 edge_hidden: int = 128,
                 linear_hidden: int = 128,
                 out_size: int = 1
                 ):
        """
        message passing neural network.

        Args:
            name (str, optional): name of this model
            node_feat (int, optional): number of input features. Defaults to 14.
            edge_feat (int, optional): number of edge features. Defaults to 3.
            gcn_size (int, optional): size of GCN embedding size. Defaults to 128.
            edge_hidden (int, optional): edge hidden embedding size. Defaults to 128.
            linear_hidden (int, optional): hidden size in fully-connected network. Defaults to 128.
            out_size (int, optional): output size. Defaults to 1.
        """
        super(MessagePassingNet, self).__init__()
        self.name = name

        self.lin0 = nn.Linear(node_feat, gcn_size)

        edge_network = nn.Sequential(
            nn.Linear(edge_feat, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, gcn_size ** 2)
        )

        self.conv = NNConv(gcn_size, gcn_size, edge_network, aggr='mean')
        self.gru = nn.GRU(input_size=gcn_size, hidden_size=gcn_size, num_layers=1)

        self.set2set = Set2Set(gcn_size, processing_steps=3)
        self.fully_connected = nn.Sequential(
            nn.Linear(2 * gcn_size, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, out_size)
        )

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            # blocks are reused here - this is subcase, generally in MPNN different instances would be used for each layer
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))

            # GRU expects
            #    input: (seq_len, batch, input_size)
            #    h0: (num_layers, batch, hidden_size)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = self.fully_connected(out)
        return out.view(-1)
