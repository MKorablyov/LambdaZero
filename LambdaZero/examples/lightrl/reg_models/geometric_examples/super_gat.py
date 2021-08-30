import torch
import torch.nn.functional as F
from torch_geometric.nn import SuperGATConv
from argparse import Namespace


class SuperGat(torch.nn.Module):
    atom_feature_extractor = True
    has_loss = True

    def __init__(self, cfg: Namespace):
        super(SuperGat, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 16)
        self.dropout = getattr(cfg, "dropout", 0.6)
        self.loss_coeff = getattr(cfg, "loss_coeff", 4.)

        self.conv1 = SuperGATConv(num_feat, dim, heads=8,
                                  dropout=self.dropout, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.conv2 = SuperGATConv(dim * 8, dim, heads=8,
                                  concat=False, dropout=self.dropout,
                                  attention_type='MX', edge_sample_ratio=0.8,
                                  is_undirected=True)

        self._loss = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        self._loss = self.loss_coeff * att_loss
        return x


network = ("SuperGat", SuperGat)
