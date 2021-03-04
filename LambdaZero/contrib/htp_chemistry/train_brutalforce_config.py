import os.path as osp
from LambdaZero.contrib.htp_chemistry.htp_env import HTP_Env_v0, HTP_Env_v1, HTP_BrutalForceTrainer
from LambdaZero.contrib.htp_chemistry.htp_env_config import htp_env_config_v0_001, htp_env_config_v1_001
from LambdaZero.contrib.loggers import log_episode_info
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

DEFAULT_CONFIG = {
    "tune_config":{
        "local_dir": summaries_dir,
    },
    "memory": 30 * 10 ** 9,
    "object_store_memory": 30 * 10 ** 9,

}

brutalforce_config_001 = {
    "tune_config": {
        "run_or_experiment": HTP_Env_v0,
        "config": htp_env_config_v0_001,
        # "checkpoint_freq": 250,
        "resources_per_trial": {"gpu": 2.0},
        "stop": {"training_iteration": 100000},
    },
}

brutalforce_config_002 = {
    "tune_config": {
        "run_or_experiment": HTP_BrutalForceTrainer,
        "config": {
            "env": HTP_Env_v1,
            "env_config": htp_env_config_v1_001,
            "num_workers": 8,
            "num_gpus_per_worker": 0.15,
            "num_gpus": 1.0,
            "callbacks": {"on_episode_end": log_episode_info},
            "framework": "torch",
            "lr": 5e-5,
            "logger_config": {
                "wandb": {
                    "project": "rlbo",
                    "api_key_file": osp.join(summaries_dir, "wandb_key")
                }}},
        "checkpoint_freq": 250,
        "stop": {"training_iteration": 100000},
    },
}