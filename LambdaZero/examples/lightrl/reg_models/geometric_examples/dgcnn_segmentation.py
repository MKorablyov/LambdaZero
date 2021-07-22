import torch
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.nn import DynamicEdgeConv
from argparse import Namespace

from LambdaZero.examples.lightrl.reg_models.geometric_examples_orig.pointnet2_classification import MLP

import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_geometric.nn import GENConv, DeepGCNLayer
from argparse import Namespace

from LambdaZero.examples.lightrl.models.utils import rsteps_rcond_conditioning


class DeeperGCNSteps(torch.nn.Module):
    atom_feature_extractor = True

    def __init__(self, cfg: Namespace):
        super(DeeperGCNSteps, self).__init__()
        num_feat = getattr(cfg, "num_feat", 14)
        self.dim = dim = hidden_channels = getattr(cfg, "dim", 64)
        num_layers = getattr(cfg, "layers", 28)
        self.dropout = getattr(cfg, "dropout", 0.0)
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
