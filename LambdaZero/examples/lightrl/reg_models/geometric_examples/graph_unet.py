import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj
from argparse import Namespace


class GUnet(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(GUnet, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropout", 0.5)
        self.dropout2 = 0.92 if self.dropout != 0 else 0

        pool_ratios = [0.5]
        self.unet = GraphUNet(num_feat, dim, dim,
                              depth=3, pool_ratios=pool_ratios)

    def forward(self, data):
        edge_index, _ = dropout_adj(data.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(data.x, p=self.dropout2, training=self.training)

        x = self.unet(x, edge_index)
        return x


network = ("GUnet", GUnet)