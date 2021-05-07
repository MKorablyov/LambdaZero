import time
from abc import ABC
import os

import numpy as np
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from torch_geometric.nn import GINEConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from LambdaZero.utils import RunningMeanStd

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
from LambdaZero.models.model_atom import MPNNet_v2

# import torch and torch.nn using ray utils
torch, nn = try_import_torch()

class GraphMolActorCritic_Generic(GraphMolActorCritic_thv1):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)
        self.max_steps = obs_space.original_space["num_steps"].n
        self.num_blocks = model_config['custom_model_config'].get("num_blocks", 135)
        self.rnd_weight = model_config['custom_model_config'].get("rnd_weight", 0)
        assert not (self.rnd_weight != 0), "NO RND implemented"
        self.rnd = False
        rnd_output_dim = model_config['custom_model_config'].get("rnd_output_dim", 1)
        self.rnd_adv_weight = model_config['custom_model_config'].get("rnd_adv_weight", 1.0)
        self.rnd_vf_loss_weight = model_config['custom_model_config'].get("rnd_vf_loss_weight", 1.0)

        self.space = obs_space.original_space['mol_graph']

        self.model = kw.get('net')(
            num_feat=self.space.num_node_feat,
            dim=kw.get('num_hidden', 64),
            num_out_per_stem=self.num_blocks
        )

        self._value_out = None
