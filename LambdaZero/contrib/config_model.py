import os.path as osp
import torch_geometric.transforms as T
import LambdaZero.inputs
import LambdaZero.utils
from LambdaZero.contrib.inputs import temp_load_data_v1
from LambdaZero.contrib.model_with_uncertainty import MolMCDropGNN
from LambdaZero.contrib.trainer import BasicTrainer
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

load_seen_config = {
    "mean": -8.6, "std": 1.1,
    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
    "raw_path": osp.join(datasets_dir, "brutal_dock/seh/raw"),
    "proc_path": osp.join(datasets_dir, "brutal_dock/seh/processed_rlbo"),
    "file_names": ["Zinc20_docked_neg_randperm_3k"],
}


model_config = {
    "train_epochs":75,
    "batch_size":75,
    "mpnn_config":{
        "drop_last":True,
        "drop_data":False,
        "drop_weights":True,
        "drop_prob":0.1,
        "num_feat":14
    },
    "lr":1e-3,
    "transform":None,
    "num_mc_samples":10,
    "log_epoch_metrics":False,
    "device":"cuda"

}

trainer_config = { # tune trainable config
    "load_data":temp_load_data_v1,
    "load_data_config":load_seen_config,
    "model": MolMCDropGNN,
    "model_config":model_config,
    "logger_config": {
        "wandb": {
            "project": "model_with_uncertainty1",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}
}

DEFAULT_CONFIG = {
    "tune_config":{
        "config": trainer_config,
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
    "tune_config":{
        "config": {"model_config": {"log_epoch_metrics":True}}
}}
