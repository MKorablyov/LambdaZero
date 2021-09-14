from argparse import Namespace

from LambdaZero.examples.gflow.trainer import basic_trainer
from LambdaZero.examples.gflow.trainer import trainer_fwdback
from LambdaZero.examples.gflow.trainer import train_with_long_credit
from LambdaZero.examples.gflow.trainer import trainer_fwdback_ppo
from LambdaZero.examples.gflow.trainer import trainer_fwdback_leafv
from LambdaZero.examples.gflow.trainer import mse_terminal
from LambdaZero.examples.gflow.trainer import gflow_credit
from LambdaZero.examples.gflow.trainer import gflow_oversample

GFLOW_TRAINERS = {
    "BasicTrainer": basic_trainer.BasicTrainer,
    "TrainGFlowFwdBack": trainer_fwdback.TrainGFlowFwdBack,
    "TrainLongCredit": train_with_long_credit.TrainLongCredit,
    "TrainGFlowFwdBackPPO": trainer_fwdback_ppo.TrainGFlowFwdBackPPO,
    "TrainGFlowFwdBackLeafV": trainer_fwdback_leafv.TrainGFlowFwdBackLeafV,
    "MseTerminal": mse_terminal.MseTerminal,
    "GflowCreditAssign": gflow_credit.GflowCreditAssign,
    "GflowOversample": gflow_oversample.GflowOversample,
}


def get_gflow_trainer(cfg: Namespace, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in GFLOW_TRAINERS,\
        "Please provide a valid GFLOW Trainer name."
    return GFLOW_TRAINERS[cfg.name](cfg, **kwargs)
