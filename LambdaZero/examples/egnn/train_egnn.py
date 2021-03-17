import os, os.path as osp
import sys

import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.tune.logger import DEFAULT_LOGGERS

from LambdaZero.utils import get_external_dirs
from LambdaZero.contrib.loggers import WandbRemoteLoggerCallback, RemoteLogger, TrialNameCreator

import LambdaZero.examples.egnn.dock_config as config
from LambdaZero.examples.egnn.dock_config import DEFAULT_CONFIG

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

config_name = sys.argv[1] if len(sys.argv) >= 2 else "egnn_000"
config = getattr(config, config_name)

config = merge_dicts(DEFAULT_CONFIG, config)

if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    os.environ['WANDB_DIR'] = summaries_dir
    os.environ["WANDB_MODE"] = "dryrun"
    remote_logger = RemoteLogger.remote()
    wandb_logger = WandbRemoteLoggerCallback(
        remote_logger=remote_logger,
        project="egnn",
        api_key_file=osp.join(summaries_dir,"wandb_key"),
        log_config=False)

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"],
                        loggers = DEFAULT_LOGGERS + (wandb_logger,),
                        trial_name_creator=TrialNameCreator(config_name)
                        )
