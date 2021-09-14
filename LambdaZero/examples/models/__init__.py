from argparse import Namespace
import torch

from LambdaZero.examples.lightrl.reg_models import mpn_model

MODELS = {
    "MPNNet": mpn_model.MPNNet,
}


def get_model(cfg: Namespace, **kwargs) -> torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in MODELS, "Please provide a valid model name."
    return MODELS[cfg.name](cfg, **kwargs)
