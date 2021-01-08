import os

from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
# from ray.tune.integration.wandb import wandb_mixin
# from ray.tune.integration.wandb import WandbLoggerCallback

import wandb

os.environ['WANDB_MODE'] = 'dryrun'

# # 1st OPTION
def train_fn(config):
    for i in range(10):
        loss = config["a"] + config["b"]
        tune.report(loss=loss)


if __name__ == "__main__":
    tune.run(
        train_fn,
        config={
            # define search space here
            "a": tune.choice([1, 2, 3]),
            "b": tune.choice([4, 5, 6]),
            # wandb configuration
            "wandb": {
                "project": "lambdazero",
                "api_key": "6b4fcbf62f4d550fbd741c7ebb3704239f200d73",
                "log_config": True
            }
        },
        loggers=DEFAULT_LOGGERS + (WandbLogger,))

# 2nd OPTION
# @wandb_mixin
# def train_fn(config):
#     for i in range(10):
#         loss = config["a"] + config["b"]
#         wandb.log({"loss": loss})
#         tune.report(loss=loss)

# tune.run(
#     train_fn,
#     config={
#         # define search space here
#         "a": tune.choice([1, 2, 3]),
#         "b": tune.choice([4, 5, 6]),
#         # wandb configuration
#         "wandb": {
#             "project": "lambdazero",
#             "api_key": "6b4fcbf62f4d550fbd741c7ebb3704239f200d73"
#         }
#     })

# # 3rd OPTION (Depreciated)
# def train_fn(config):
#     for i in range(10):
#         loss = config["a"] + config["b"]
#         tune.report(loss=loss)
#
# tune.run(
#     train_fn,
#     config={
#         # define search space here
#         "a": tune.choice([1, 2, 3]),
#         "b": tune.choice([4, 5, 6])
#     },
#     callbacks=[WandbLoggerCallback(
#         project="lz_test",
#         api_key="6b4fcbf62f4d550fbd741c7ebb3704239f200d73",
#         log_config=True)]
# )

