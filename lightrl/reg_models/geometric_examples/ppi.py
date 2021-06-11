import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from argparse import Namespace


class PPINet(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(PPINet, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 256)
        self.dropout = getattr(cfg, "dropout", 0.5)

        self.conv1 = GATConv(num_feat, dim, heads=4)
        self.lin1 = torch.nn.Linear(num_feat, 4 * dim)
        self.conv2 = GATConv(4 * dim, dim, heads=4)
        self.lin2 = torch.nn.Linear(4 * dim, 4 * dim)
        self.conv3 = GATConv(4 * dim, dim, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * dim, dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


network = ("PPINet", PPINet)
