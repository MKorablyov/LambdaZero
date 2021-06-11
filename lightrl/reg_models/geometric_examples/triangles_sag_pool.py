import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv, GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool
from torch_scatter import scatter_mean
from argparse import Namespace


class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = torch.softmax(data.x, dim=0).flatten()
        data.x = None
        return data


class TriangSag(torch.nn.Module):
    atom_feature_extractor = False
    has_loss = True

    def __init__(self, cfg: Namespace):
        super(TriangSag, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 64)
        self.dropout = getattr(cfg, "dropout", 0.5)
        self.loss_coeff = getattr(cfg, "loss_coeff", 100)
        self.num_out = num_out = getattr(cfg, "num_out", 1)

        self.transform = T.Compose([HandleNodeAttention(), T.OneHotDegree(max_degree=num_feat)])

        self.conv1 = GINConv(Seq(Lin(num_feat+1, dim), ReLU(), Lin(dim, dim)))
        self.pool1 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.conv2 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))
        self.pool2 = SAGPooling(dim, min_score=0.001, GNN=GCNConv)
        self.conv3 = GINConv(Seq(Lin(dim, dim), ReLU(), Lin(dim, dim)))

        self.lin = torch.nn.Linear(dim, num_out)
        self._loss = None
        self._per_atom_out_size = self.dim

    def forward(self, data):
        data = self.transform(data)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(
            x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool2(
            x, edge_index, None, batch)
        ratio = x.size(0) / data.x.size(0)
        x = per_atom_out = F.relu(self.conv3(x, edge_index))

        x = global_max_pool(x, batch)

        x = self.lin(x)

        attn_loss = F.kl_div(
            torch.log(score + 1e-14), data.attn[perm], reduction='none')
        attn_loss = scatter_mean(attn_loss, batch)
        self._loss = (self.loss_coeff * attn_loss).mean()

        if self.num_out == 1:
            x = x.view(-1)

        return x, per_atom_out

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size


network = ("TriangSag", TriangSag)