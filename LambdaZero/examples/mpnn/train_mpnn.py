import os
import sys
import os.path as osp
import torch

import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.tune.logger import DEFAULT_LOGGERS

from LambdaZero.utils import get_external_dirs, BasicRegressor
from LambdaZero.utils import train_epoch, eval_epoch

import LambdaZero.inputs
import LambdaZero.models
from LambdaZero.contrib.loggers import WandbRemoteLoggerCallback, RemoteLogger, TrialNameCreator
from LambdaZero.examples.mpnn import config

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

config_name = sys.argv[1] if len(sys.argv) >= 2 else "mpnn000"
config = getattr(config, config_name)

transform = LambdaZero.utils.Complete()

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "target": "dockscore",
        "target_norm": [-8.6597, 1.0649],
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
        "batch_size": 96,
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/seh"),
            "props": ["dockscore", "smiles"],
            "transform": transform,
            "file_names": ["Zinc20_docked_neg_randperm_3k"],
        },
        "model": LambdaZero.models.MPNNet,
        "model_config": {},
        "optimizer": {
            "type": torch.optim.Adam,
            "config": {
                "lr": 0.001
            }
        },
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },
    "summaries_dir": summaries_dir,
    "memory": 50 * 10 ** 9,
    "object_store_memory": 50 * 10 ** 9,

    "stop": {"training_iteration": 500},
    "resources_per_trial": {
        "cpu": 8,  # fixme - calling ray.remote would request resources outside of tune allocation
        "gpu": 2.0
    },
    "keep_checkpoint_num": 2,
    "checkpoint_score_attr": "train_loss",
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 25,
}


config = merge_dicts(DEFAULT_CONFIG, config)


if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    os.environ['WANDB_DIR'] = summaries_dir
    os.environ["WANDB_MODE"] = "dryrun"
    remote_logger = RemoteLogger.remote()
    wandb_logger = WandbRemoteLoggerCallback(
        remote_logger=remote_logger,
        project="egnn",
        api_key_file=osp.join(summaries_dir, "wandb_key"),
        log_config=False)
    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"],
                        loggers=DEFAULT_LOGGERS + (wandb_logger,),
                        trial_name_creator=TrialNameCreator(config_name)
                        )
