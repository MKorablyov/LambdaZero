import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
from argparse import Namespace

DEG = [     1,     72,    201,    816,   1790,   3756,   6923,  12768,  20286,
         31710,  51623,  82296, 124280, 177576, 251115, 326064, 395760, 456840,
        506179, 516200, 507003, 493746, 489256, 453936, 420025, 411320, 427761,
        420700, 420500, 426780, 414284, 407008, 394053, 360910, 322245, 313704,
        282902, 270940, 237783, 209000, 193766, 177870, 162110, 144848, 121230,
        112700,  93483,  88512,  72275,  80700,  68799,  56784,  42665,  30996,
         25630,  12936,   9804,   8584,   5251,   3480,   3111,   2728,   1890,
          1472,   1235,    330,    201,     68,     69,      0,     71]


class PNA(torch.nn.Module):
    atom_feature_extractor = False

    def __init__(self, cfg: Namespace):
        super(PNA, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = getattr(cfg, "dim", 75)
        edge_attr_dim = getattr(cfg, "edge_attr_dim", 4)
        edge_dim = getattr(cfg, "edge_dim", 50)

        self.num_out = num_out = getattr(cfg, "num_out", 1)

        deg = getattr(cfg, "deg", DEG)
        deg = torch.LongTensor(deg)

        # self.node_emb = Embedding(num_feat, dim) # TODO What ?
        # self.edge_emb = Embedding(edge_attr_dim, edge_dim)
        self.node_emb = torch.nn.Linear(num_feat, dim)  # TODO What ?
        self.edge_emb = torch.nn.Linear(edge_attr_dim, edge_dim)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        tower_num = 4
        while dim % tower_num != 0:
            tower_num += 1

        for _ in range(4):
            conv = PNAConv(in_channels=dim, out_channels=dim,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=edge_dim, towers=tower_num, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(dim))

        self.mlp = Sequential(Linear(dim, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, num_out))

        self._per_atom_out_size = dim

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        per_atom_out = x
        x = global_add_pool(x, batch)
        x = self.mlp(x)

        if self.num_out == 1:
            x = x.view(-1)

        return x, per_atom_out

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size


network = ("PNA", PNA)


