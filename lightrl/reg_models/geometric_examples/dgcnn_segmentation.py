import torch
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.nn import DynamicEdgeConv
from argparse import Namespace

from lightrl.reg_models.geometric_examples_orig.pointnet2_classification import MLP


class DGCNN(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace, k=30, aggr='max'):
        super(DGCNN, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 64)
        self.dropout = getattr(cfg, "dropout", 0.5)

        self.conv1 = DynamicEdgeConv(MLP([2 * num_feat, dim, dim]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * dim, dim, dim]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * dim, dim, dim]), k, aggr)
        self.lin1 = MLP([3 * dim, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(self.dropout), MLP([256, 128]),
                       Dropout(self.dropout), Lin(128, dim))

    def forward(self, data):
        x, batch = data.x, data.batch
        x0 = x
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return out


network = ("DGCNN", DGCNN)
