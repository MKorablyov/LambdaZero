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
from LambdaZero.examples.egnn.egnn import EGNNet
import LambdaZero.examples.egnn.qm9_config as config
from LambdaZero.contrib.loggers import WandbRemoteLoggerCallback, RemoteLogger

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

config_name = sys.argv[1] if len(sys.argv) >= 2 else "qm9_all"
config = getattr(config, config_name)

transform = LambdaZero.utils.Complete()

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "target": "gridscore",
        "target_norm": [-49.4, 7.057],
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
        "batch_size": 96,
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore"],
            "transform": transform,
            "file_names": ["Zinc15_260k_0"] #, "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
        },
        "model": EGNNet,
        "model_config": {
            "n_layers": 7,
            "feats_dim": 14,
            "pos_dim": 3,
            "edge_attr_dim": 0,
            "m_dim": 128,
        },
        "optimizer": {
            "type": torch.optim.Adam,
            "config": {
                "weight_decay": 1e-16,
                "lr": 5e-4,
            }
        },
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },
    "summaries_dir": summaries_dir,
    "memory": 30 * 10 ** 9,
    "object_store_memory": 30 * 10 ** 9,

    "stop": {"training_iteration": 1000},
    "resources_per_trial": {
        "cpu": 8,
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
    wandb_callback = WandbRemoteLoggerCallback(
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
                        loggers = DEFAULT_LOGGERS + (wandb_callback,),)
