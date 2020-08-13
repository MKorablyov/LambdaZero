import ray
import torch
from ray.tune import tune

from LambdaZero.examples.env3d.dataset import ENV3D_DATA_PROPERTIES
from LambdaZero.examples.env3d.dataset.processing import (
    env3d_proc,
    transform_concatenate_positions_to_node_features,
)
from LambdaZero.examples.env3d.env3d_model_trainer import Env3dModelTrainer
from LambdaZero.examples.env3d.epoch_stepper import train_epoch, eval_epoch
from LambdaZero.examples.env3d.models.joint_prediction_model import BlockAngleModel
from LambdaZero.examples.env3d.parameter_inputs import get_input_arguments, read_configuration_file
from LambdaZero.examples.env3d.wandb_logger import LambdaZeroWandbLogger
from LambdaZero.inputs import BrutalDock
from LambdaZero.utils import get_external_dirs

_, _, summaries_dir = get_external_dirs()

if __name__ == "__main__":
    args = get_input_arguments()

    input_config = read_configuration_file(args.config)

    wandb_logging_config = dict(entity="lambdazero",
                                project=input_config["logging"].get("project", "env3d"),
                                name=input_config["logging"].get("name", "debug")
                                )

    ray.init()

    env3d_config = {
        "trainer": Env3dModelTrainer,
        "trainer_config": {
            "monitor": True,
            "env_config": {"wandb": wandb_logging_config},
            "dataset": BrutalDock,
            "seed_for_dataset_split": input_config["trainer_config"]["seed_for_dataset_split"],
            "train_ratio": input_config["trainer_config"]["train_ratio"],
            "valid_ratio": input_config["trainer_config"]["valid_ratio"],
            "batchsize": input_config["trainer_config"]["batch_size"],
            "model": BlockAngleModel,
            "model_config": input_config["model_config"],
            "optimizer": torch.optim.Adam,
            "optimizer_config": {"lr": input_config["trainer_config"]["learning_rate"]},
            "loss_mode": input_config["trainer_config"]["loss_mode"],
            "angle_loss_weight": input_config["trainer_config"]["angle_loss_weight"],
            "block_loss_weight": input_config["trainer_config"]["block_loss_weight"],
            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
            "patience": input_config["trainer_config"]["patience"],
            "dataset_config": {
                "root": args.root_path,
                "props": ENV3D_DATA_PROPERTIES,
                "proc_func": env3d_proc,
                "transform": transform_concatenate_positions_to_node_features,
                "file_names": [args.data_file_name.replace(".feather", "")], # remove suffix in case it is there
            },
        },
        "summaries_dir": summaries_dir,
        "stop": {"training_iteration": input_config["trainer_config"]["num_epochs"],
            "early_stopping": True},
        "resources_per_trial": input_config["resources_per_trial"],
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
