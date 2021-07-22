from argparse import Namespace

from LambdaZero.examples.gflow.trainer import baseline_trainer
from LambdaZero.examples.gflow.trainer import trainer_v2
from LambdaZero.examples.gflow.trainer import trainer_fwdback

GFLOW_TRAINERS = {
    "TrainGFlow": baseline_trainer.TrainGFlow,
    "TrainGFlowV2": trainer_v2.TrainGFlowV2,
    "TrainGFlowFwdBack": trainer_fwdback.TrainGFlowFwdBack
}


def get_gflow_trainer(cfg: Namespace, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in GFLOW_TRAINERS,\
        "Please provide a valid GFLOW Trainer name."
    return GFLOW_TRAINERS[cfg.name](cfg, **kwargs)
