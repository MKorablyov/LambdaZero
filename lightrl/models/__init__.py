import torch
from argparse import Namespace
import gym

from . import model_atom
from . import model_test_multihead
from . import model_egnn

MODELS = {
    "MPNNet_v2": model_atom.MPNNet_v2,
    "MPNNetMultihead": model_test_multihead.MPNNetMultihead,
    "EGNNetRLBO": model_egnn.EGNNetRLBO,
}


def get_model(cfg: Namespace, obs_space: dict, action_space: gym.spaces, **kwargs) -> \
        torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in MODELS, "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)
