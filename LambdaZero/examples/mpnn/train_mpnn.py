import os
import sys
import os.path as osp
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
        "target": "gridscore",
        "target_norm": [-49.4, 7.057],
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
        "batch_size": 16,  # 25
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore"],
            "transform": transform,
            "file_names": ["Zinc15_260k_0"],#, "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
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
