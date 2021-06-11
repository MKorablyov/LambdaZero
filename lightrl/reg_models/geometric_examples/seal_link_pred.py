# Paper: Link Prediction Based on Graph Neural Networks (NeurIPS 2018)
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d

from torch_geometric.nn import GCNConv, global_sort_pool
from argparse import Namespace


class DGCNNSeal(torch.nn.Module):
    atom_feature_extractor = False

    def __init__(self, cfg: Namespace, GNN=GCNConv, k=32):
        super(DGCNNSeal, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = hidden_channels = getattr(cfg, "dim", 32)
        num_layers = getattr(cfg, "layers", 3)
        self.dropout = getattr(cfg, "dropout", 0.2)

        self.num_out = num_out = getattr(cfg, "num_out", 1)

        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(num_feat, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        self._per_atom_out_size = total_latent_dim

        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, num_out)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = per_atom_out = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.num_out == 1:
            x = x.view(-1)

        return x, per_atom_out

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size


network = ("DGCNNSeal", DGCNNSeal)
