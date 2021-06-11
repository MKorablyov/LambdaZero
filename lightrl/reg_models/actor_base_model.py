from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
import torch


class ActorWrapper(nn.Module):
    is_actor = True

    def __init__(self, model_base, cfg: Namespace):
        super().__init__()

        num_out_per_stem = getattr(cfg, "num_out_per_stem", 105)

        self._model = model_base
        dim = self._model.per_atom_out_size

        # --> Change 3 Add new ln
        self.node2stem = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, num_out_per_stem))
        self.node2jbond = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(inplace=True), nn.Linear(dim, 1))

    def forward(self, data):
        scalar_outs, per_atom_out = self._model(data)

        data.stem_preds = self.node2stem(per_atom_out[data.stem_atmidx])

        data.jbond_preds = self.node2jbond(per_atom_out[data.jbond_atmidx.flatten()])

        # mean pooling of the 2 jbond atom preds
        data.jbond_preds = data.jbond_preds.reshape((data.jbond_atmidx.shape)).mean(1)

        stop_logit = scalar_outs[:, 1:2]
        break_logits = data.jbond_preds.reshape((data.num_graphs, -1))
        add_logits = data.stem_preds.reshape((data.num_graphs, -1))

        actor_logits = torch.cat([stop_logit, break_logits, add_logits], 1)

        value = scalar_outs[:, :1]
        return value, actor_logits