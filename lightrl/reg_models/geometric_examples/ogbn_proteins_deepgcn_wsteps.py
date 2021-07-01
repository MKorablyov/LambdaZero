import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_geometric.nn import GENConv, DeepGCNLayer
from argparse import Namespace

from lightrl.models.utils import rsteps_rcond_conditioning


class DeeperGCNSteps(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(DeeperGCNSteps, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = hidden_channels = getattr(cfg, "dim", 64)
        num_layers = getattr(cfg, "layers", 28)
        self.dropout = getattr(cfg, "dropout", 0.1)
        edge_attr_dim = getattr(cfg, "edge_attr_dim", 4)
        self._conditioning = {x: y for x, y in getattr(cfg, "conditioning", [["r_steps", 10], ["rcond", 1]])}
        self._cond_level = cond_level = getattr(cfg, "cond_level", 14)
        self.cond_size = cond_size = sum(self._conditioning.values())
        if cond_size > 0:
            assert 1 <= self._cond_level < num_layers + 1, "Lower level cond"
        else:
            self._cond_level = 0

        self.node_encoder = Linear(num_feat, hidden_channels)
        self.edge_encoder = Linear(edge_attr_dim, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if cond_size > 0 and i >= self._cond_level:
                crt_h = hidden_channels + cond_size
            else:
                crt_h = hidden_channels

            conv = GENConv(crt_h, crt_h, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels + cond_size if i == 1 else crt_h, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=self.dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(crt_h, dim)

    def forward(self, inputs):
        cond_level = self._cond_level - 1
        data = inputs.mol_graph

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        if cond_level == 0:
            x = rsteps_rcond_conditioning(x, inputs, max_steps=self._conditioning["r_steps"])
            edge_attr = torch.cat([
                edge_attr, torch.ones(edge_attr.size(0), self.cond_size, device=edge_attr.device)
            ], dim=1)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        lvl = 1
        for layer in self.layers[1:]:
            if cond_level == lvl:
                x = rsteps_rcond_conditioning(x, inputs, max_steps=self._conditioning["r_steps"])
                edge_attr = torch.cat([
                    edge_attr,
                    torch.ones(edge_attr.size(0), self.cond_size, device=edge_attr.device)
                ], dim=1)

            x = layer(x, edge_index, edge_attr)
            lvl += 1

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return x


network = ("DeeperGCNSteps", DeeperGCNSteps)