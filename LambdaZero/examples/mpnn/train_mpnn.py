import os
import sys
import torch

import ray
from ray import tune
from ray.rllib.utils import merge_dicts

from LambdaZero.utils import get_external_dirs, BasicRegressor
from LambdaZero.utils import train_epoch, eval_epoch

import LambdaZero.inputs
import LambdaZero.models
from LambdaZero.examples.mpnn import config

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

config_name = sys.argv[1] if len(sys.argv) >= 2 else "mpnn000"
config = getattr(config, config_name)

transform = LambdaZero.utils.Complete()

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "target": "dockscore",
        "target_norm": [-8.6, 1.1],
        "dataset_split_path": os.path.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
        "batch_size": 32,  # 25
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/seh"),
            "props": ["dockscore"],
            "transform": transform,
            "file_names": ["Zinc20_docked_neg_randperm_3k"],
        },
        "model": LambdaZero.models.MPNNet,
        "model_config": {},
        "optimizer": {
            "type": torch.optim.Adam,
            "config": {
                "lr": 1e-3
            }
        },
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },
    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,

    "stop": {"training_iteration": 200},
    "resources_per_trial": {
        "cpu": 4,  # fixme - calling ray.remote would request resources outside of tune allocation
        "gpu": 1.0
    },
    "keep_checkpoint_num": 2,
    "checkpoint_score_attr": "train_loss",
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 25,
}


config = merge_dicts(DEFAULT_CONFIG, config)


if __name__ == "__main__":
    ray.init(_memory=config["memory"])
    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"])
