from typing import Dict

from torch import nn

import wandb

from LambdaZero.datasets.brutal_dock.loggers.experiment_logger import ExperimentLogger

ENTITY = "lambdazero"  # this is the name of the group in our wandb account


class WandbLogger(ExperimentLogger):
    def __init__(self, run_parameters: Dict[str, str], tracking_uri: str, tags: Dict[str, str], notes: str = None):
        """
        Class to manage the Weights-and-Biases logger.

        Ignore the tags.
        """
        super().__init__(run_parameters, tracking_uri, tags)

        experiment_name = run_parameters.pop("experiment_name", 'none')
        run_name = run_parameters.pop("run_name", 'none')

        wandb.init(project=experiment_name,
                   entity=ENTITY,
                   name=run_name,
                   dir=tracking_uri,
                   notes=notes,
                   tags=run_parameters
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
