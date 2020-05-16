from typing import Dict

from torch import nn

import wandb

ENTITY = "lambdazero"  # this is the name of the group in our wandb account


class WandbLogger:
    def __init__(self, experiment_name: str, tracking_uri: str, notes: str = None):
        """
        Class to manage the Weights-and-Biases logger.
        Args:
            experiment_name (str): The name of the experiment
            kwargs : dictionary of parameters for specific logger.
        """
        super().__init__()

        wandb.init(project="first-steps",
                   entity=ENTITY,
                   name=experiment_name,
                   dir=tracking_uri,
                   notes=notes
                   )

    def watch_model(self, model: nn.Module):
        wandb.watch(model)

    def log_metrics(self, key: str, value: float):
        logging_dict = {key: value}
        wandb.log(logging_dict)
        pass

    def log_configuration(self, config: Dict):
        wandb.config.update(config)

    def log_artifact(self, path: str):
        wandb.save(path)

    def finalize(self):
        pass
