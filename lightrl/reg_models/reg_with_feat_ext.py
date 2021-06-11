from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set


class RegBase(nn.Module):
    def __init__(self, cfg: Namespace, feature_extractor: nn.Module):
        super(RegBase, self).__init__()
        self._feature_extractor = feature_extractor(cfg)
        dim = self._feature_extractor.dim
        processing_steps = getattr(cfg, "processing_steps", 3)
        self.num_out = num_out = getattr(cfg, "num_out", 1)

        self._per_atom_out_size = dim

        self.set2set = Set2Set(dim, processing_steps=processing_steps)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, num_out)

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size

    def forward(self, data):
        feat = per_atom_out = self._feature_extractor(data)

        out = self.set2set(feat, data.batch)
        out = nn.functional.leaky_relu(self.lin1(out))
        out = self.lin2(out)

        if self.num_out == 1:
            out = out.view(-1)

        return out, per_atom_out
