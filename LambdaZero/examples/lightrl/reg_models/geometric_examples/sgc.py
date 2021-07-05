import torch
from torch_geometric.nn import SGConv
from argparse import Namespace


class SGC(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(SGC, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)

        self.conv1 = SGConv(num_feat, dim, K=2, cached=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return x


network = ("SGC", SGC)