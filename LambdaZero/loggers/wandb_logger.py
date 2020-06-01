import os
from typing import Dict, List

import wandb
from torch import nn

from LambdaZero.loggers.experiment_logger import ExperimentLogger

ENTITY = "lambdazero"  # this is the name of the group in our wandb account


def set_wandb_to_dryrun():
    os.environ["WANDB_MODE"] = "dryrun"


class WandbLogger(ExperimentLogger):
    """
    Class to manage the Weights-and-Biases logger.
    """
    def __init__(self, run_parameters: Dict[str, str], tracking_uri: str,  execution_filename: str,
                 notes: str = None, tags: List[str] = None):
        super().__init__(run_parameters, tracking_uri, execution_filename)

        wandb.init(project=self.experiment_name,
                   entity=ENTITY,
                   name=self.run_name,
                   dir=self.tracking_uri,
                   notes=notes,
                   tags=tags
                   )

    def watch_model(self, model: nn.Module):
        wandb.watch(model)

    def _log_metrics(self, key: str, value: float, step: int):
        logging_dict = {key: value}
        wandb.log(logging_dict, step=step)

    def log_configuration(self, config: Dict):
        wandb.config.update(config)

    def log_artifact(self, path: str):
        wandb.save(path)

    def finalize(self):
        pass
