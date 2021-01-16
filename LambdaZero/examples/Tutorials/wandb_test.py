# After your job is finished, use this command from the login node to sync the logs:
# wandb sync wandb/<log_dir>


import os

from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
# from ray.tune.integration.wandb import WandbLoggerCallback

import wandb

os.environ['WANDB_MODE'] = 'dryrun'

def train_fn(config):
    for i in range(10):
        loss = config["a"] + config["b"]
        tune.report(loss=loss)

tune.run(
    train_fn,
    config={
        # define search space here
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6])
    },
    callbacks=[WandbLoggerCallback(
        project="lambdazero",
        api_key="6b4fcbf62f4d550fbd741c7ebb3704239f200d73",
        log_config=True)]
)

