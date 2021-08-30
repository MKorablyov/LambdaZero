import torch
from torch_geometric.nn import GATConv

from argparse import Namespace


class Breadth(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index))
        return x


class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):
    def __init__(self, in_dim, dim, lstm_hidden):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, dim)
        self.depth_func = Depth(dim, lstm_hidden)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)


class GeniePath(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(GeniePath, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 256)
        self.lstm_hidden = lstm_hidden = dim  # Default 256
        layer_num = 4

        self.lstm_hidden = lstm_hidden
        self.lin1 = torch.nn.Linear(num_feat, dim)
        self.gplayers = torch.nn.ModuleList(
            [GeniePathLayer(dim, dim, lstm_hidden) for i in range(layer_num)])
        self.lin2 = torch.nn.Linear(dim, dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        for i, l in enumerate(self.gplayers):
            x, (h, c) = self.gplayers[i](x, edge_index, h, c)
        x = self.lin2(x)
        return x


class GeniePathLazy(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(GeniePathLazy, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 256)
        self.lstm_hidden = lstm_hidden = dim  # Default 256
        layer_num = 4

        self.lin1 = torch.nn.Linear(num_feat, dim)
        self.breadths = torch.nn.ModuleList(
            [Breadth(dim, dim) for i in range(layer_num)])
        self.depths = torch.nn.ModuleList(
            [Depth(dim * 2, lstm_hidden) for i in range(layer_num)])
        self.lin2 = torch.nn.Linear(dim, dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        h_tmps = []
        for i, l in enumerate(self.breadths):
            h_tmps.append(self.breadths[i](x, edge_index))
        x = x[None, :]
        for i, l in enumerate(self.depths):
            in_cat = torch.cat((h_tmps[i][None, :], x), -1)
            x, (h, c) = self.depths[i](in_cat, h, c)
        x = self.lin2(x[0])
        return x


network = [("GeniePathLazy", GeniePathLazy), ("GeniePath", GeniePath)]
