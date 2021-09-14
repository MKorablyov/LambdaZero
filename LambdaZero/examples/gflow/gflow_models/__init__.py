from argparse import Namespace
import torch

from LambdaZero.examples.gflow.gflow_models import gflow1_model_atom, gflow1_model_block
from LambdaZero.examples.gflow.gflow_models import deepgcn
from LambdaZero.examples.gflow.gflow_models import gflow1_model_block_multi

GFLOW_MODELS = {
    "GraphAgent": gflow1_model_block.GraphAgent,
    "MolACGCN": gflow1_model_atom.MolAC_GCN,
    "DeeperGCNv0": deepgcn.DeeperGCNGflow,
    "GraphAgentMultiV": gflow1_model_block_multi.GraphAgentMultiV,
}


def get_gflow_model(cfg: Namespace, **kwargs) -> torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in GFLOW_MODELS,\
        "Please provide a valid GFLOW model name."
    return GFLOW_MODELS[cfg.name](cfg, **kwargs)
