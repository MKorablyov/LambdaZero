import os.path as osp
#import torch_geometric.transforms as T
import LambdaZero.inputs
import LambdaZero.utils
from LambdaZero.contrib.data import temp_load_data, config_temp_load_data_v1
from LambdaZero.contrib.modelBO import MolMCDropGNN, config_MolMCDropGNN_v1
from LambdaZero.contrib.trainer import BasicTrainer
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


config_modelBO_run_v1 = { # tune trainable config
    "load_data":temp_load_data,
    "load_data_config":config_temp_load_data_v1,
    "model": MolMCDropGNN,
    "model_config": config_MolMCDropGNN_v1,
    "logger_config": {
        "wandb": {
            "project": "model_with_uncertainty1",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}
}

config_modelBO_v1 = {
    "tune_config":{
        "config": config_modelBO_run_v1,
        "local_dir": summaries_dir,
        "run_or_experiment": BasicTrainer,
        "checkpoint_freq": 250000, # todo: implement _save to be able to checkpoint
        "stop":{"training_iteration": 1},
        "resources_per_trial": {
            "cpu": 4,
            "gpu": 1.0
        },
    },
    "memory": 10 * 10 ** 9,
    "object_store_memory": 10 * 10 ** 9
}

model_001 = {
    "default": config_modelBO_v1,
    "tune_config":{
        "config": {"model_config": {"log_epoch_metrics":True,
                                    "train_epochs":75
                                    }}
}}