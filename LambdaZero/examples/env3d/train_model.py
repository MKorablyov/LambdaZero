import ray
import torch
from ray import tune

from LambdaZero.examples.env3d.dataset import ENV3D_DATA_PROPERTIES
from LambdaZero.examples.env3d.dataset.processing import (
    env3d_proc,
    transform_concatenate_positions_to_node_features,
)
from LambdaZero.examples.env3d.env3d_model_trainer import Env3dModelTrainer
from LambdaZero.examples.env3d.epoch_stepper import train_epoch, eval_epoch
from LambdaZero.examples.env3d.models.joint_prediction_model import BlockAngleModel
from LambdaZero.examples.env3d.wandb_logger import LambdaZeroWandbLogger
from LambdaZero.inputs import BrutalDock
from LambdaZero.utils import get_external_dirs

_, _, summaries_dir = get_external_dirs()

wandb_logging_config = dict(project="env3d", entity="lambdazero", name="debug3")


if __name__ == "__main__":

    ray.init(local_mode=True)
    # ray.init(memory=env3d_config["memory"])

    env3d_config = {
        "trainer": Env3dModelTrainer,
        "trainer_config": {
            "monitor": True,
            "env_config": {"wandb": wandb_logging_config},
            "dataset": BrutalDock,
            "seed_for_dataset_split": 0,
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "batchsize": 5,
            "model": BlockAngleModel,  # to do: insert a real model here
            "model_config": {},
            "optimizer": torch.optim.Adam,
            "optimizer_config": {"lr": 1e-3},
            "angle_loss_weight": 1,
            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
            "dataset_config": {
                "root": "/Users/bruno/LambdaZero/summaries/env3d/dataset/from_cluster/RUN4/combined",
                "props": ENV3D_DATA_PROPERTIES,
                "proc_func": env3d_proc,
                "transform": transform_concatenate_positions_to_node_features,
                "file_names": ["debug"],
            },
        },
        "summaries_dir": summaries_dir,
        "memory": 10 * 10 ** 9,
        "stop": {"training_iteration": 200},
        "resources_per_trial": {
            "cpu": 1,  # fixme - calling ray.remote would request resources outside of tune allocation
            "gpu": 0.0,
        },
        "keep_checkpoint_num": 2,
        "checkpoint_score_attr": "train_loss",
        "num_samples": 1,
        "checkpoint_at_end": False,
    }

    analysis = tune.run(
        env3d_config["trainer"],
        config=env3d_config["trainer_config"],
        stop=env3d_config["stop"],
        resources_per_trial=env3d_config["resources_per_trial"],
        num_samples=env3d_config["num_samples"],
        checkpoint_at_end=env3d_config["checkpoint_at_end"],
        local_dir=summaries_dir,
        checkpoint_freq=env3d_config.get("checkpoint_freq", 1),
        loggers=[LambdaZeroWandbLogger],
    )
