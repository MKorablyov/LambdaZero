import os
from typing import Dict, List

import wandb
from torch import nn

from LambdaZero.loggers.experiment_logger import ExperimentLogger
from LambdaZero.loggers.ray_tune_logger import RayTuneLogger

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


class RayTuneWandbLogger(RayTuneLogger):
    """
    Class to manage the Weights-and-Biases logger within a ray.tune trainable object.
    The initialisation follows the instructions here:
    <https://docs.ray.io/en/master/tune/api_docs/logging.html?highlight=tune%20logger#trainable-logging>
    """

    def __init__(self, config: Dict[str, str], log_dir: str, trial_id: str):

        super().__init__(config, log_dir, trial_id)

        wandb.init(project=config["experiment_name"],
                   entity=ENTITY,
                   name=trial_id,
                   id=trial_id,
                   reinit=True,
                   resume=trial_id,
                   dir=log_dir,
                   allow_val_change=True,
                   )

        wandb.config.update(config, allow_val_change=True)

    def log_metrics(self, result_dict: Dict, step: int):
        wandb.log(result_dict, step=step)
