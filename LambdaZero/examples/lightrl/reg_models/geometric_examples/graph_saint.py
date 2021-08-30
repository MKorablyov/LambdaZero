import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from argparse import Namespace


class Saint(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(Saint, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = hidden_channels= getattr(cfg, "dim", 256)
        self.dropout = getattr(cfg, "dropout", 0.2)
        use_normalization = getattr(cfg, "use_normalization", False)

        in_channels = num_feat
        out_channels = dim
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

        self.set_aggr('add' if use_normalization else 'mean')

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, data):
        x0, edge_index, edge_weight = data.x, data.edge_index, getattr(data, "edge_weight", None)
        p = self.dropout

        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=p, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=p, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=p, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x


network = ("Saint", Saint)
