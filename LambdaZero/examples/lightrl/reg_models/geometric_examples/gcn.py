import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from argparse import Namespace

"""
Why data.edge_attr needs to be 1 dimensional?!
"""

# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
def feat_data(data):
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class GCN(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(GCN, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropoutdropout", 0.5)
        use_gdc = getattr(cfg, "use_gdc", False)

        self.conv1 = GCNConv(num_feat, dim, cached=False, normalize=not use_gdc)
        self.conv2 = GCNConv(dim, dim, cached=False, normalize=not use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        edge_weight = None  # GCNConv doesn't work with our edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


network = ("GCN", GCN)
