import torch
from argparse import Namespace
import glob
from importlib import import_module

from . import mpn_model
from . import model_reg_atom
from . import model_egnn
from .reg_with_feat_ext import RegBase
from .actor_base_model import ActorWrapper


MODELS = {
    "MPNNet": mpn_model.MPNNet,
    "MPNNet_v2": model_reg_atom.MPNNet_v2,
    "EGNNetRLBO": model_egnn.EGNNetRLBO,
}


def get_model(cfg: Namespace, **kwargs) -> torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in MODELS, "Please provide a valid model name."
    return MODELS[cfg.name](cfg, **kwargs)


def get_reg_model(cfg: Namespace, **kwargs) -> torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in MODELS, "Please provide a valid model name."

    if hasattr(MODELS[cfg.name], "atom_feature_extractor") and \
            MODELS[cfg.name].atom_feature_extractor:
        return RegBase(cfg, MODELS[cfg.name])
    else:
        return MODELS[cfg.name](cfg, **kwargs)


def get_actor_model(cfg: Namespace, **kwargs) -> torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in MODELS, "Please provide a valid model name."

    if getattr(MODELS[cfg.name], "is_actor", False):
        return MODELS[cfg.name](cfg, **kwargs)

    if getattr(MODELS[cfg.name], "atom_feature_extractor", False):
        model_base = RegBase(cfg, MODELS[cfg.name])
    else:
        assert getattr(MODELS[cfg.name], "per_atom_output", True), f"No per atom features {cfg.name}"
        model_base = MODELS[cfg.name](cfg, **kwargs)

    return ActorWrapper(model_base, cfg, **kwargs)


# ADD batch of networks
python_paths = glob.glob("lightrl/reg_models/geometric_examples/*.py")
var_name = "network"

for python_path in python_paths:
    module_path = python_path.replace("/", ".").replace(".py", "")

    mod = import_module(module_path)
    if hasattr(mod, var_name):

        if isinstance(getattr(mod, var_name), list):
            _models = getattr(mod, var_name)
        else:
            _models = [getattr(mod, var_name)]

        for network_name, network in _models:
            assert network_name not in MODELS, f"Network name already exists {network_name} - " \
                                               f"{MODELS[network_name]} was before {network}"
            MODELS[network_name] = network
