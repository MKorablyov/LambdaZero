import os.path as osp
from ray.rllib.utils import merge_dicts
import LambdaZero.utils
from LambdaZero.contrib.trainer import RandomPolicyTrainer
from LambdaZero.contrib.loggers import log_episode_info
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyReward, ProxyRewardSparse
from LambdaZero.contrib.oracle import config_DockingOracle_v1
from LambdaZero.contrib.proxy import config_ProxyUCB_v2
from LambdaZero.environments import BlockMolEnvGraph_v1


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

# change default settings for oracle
config_DockingOracle_def1 = merge_dicts(config_DockingOracle_v1, {"num_threads": 24})

config_randpolicy_run_v1 = { # tune trainable config to be more precise
    "env": BlockMolEnvGraph_v1,
    "env_config": {
        "molMDP_config": {"blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json")},
        "random_steps": 4,
        "allow_removal": True,
        "reward": ProxyReward,
        "reward_config": {
            "synth_options":{"num_gpus":0.05},
            "qed_cutoff": [0.2, 0.5],
            "synth_cutoff":[0, 4],
            "scoreProxy":ProxyUCB,
            "scoreProxy_config":config_ProxyUCB_v2,
            "scoreProxy_options":{"num_cpus":1, "num_gpus":1.0},
            "actor_sync_freq": 1500,
        },

    },
    "num_workers": 8,
    "num_gpus_per_worker": 0.15,
    "num_gpus": 0.15,
    "callbacks": {"on_episode_end": log_episode_info},
    "framework": "torch",
    "lr": 5e-5,
    "logger_config":{
        "wandb": {
            "project": "rlbo4",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}}


config_randpolicy_v1 = {
    "tune_config":{
        "config":config_randpolicy_run_v1,
        "local_dir": summaries_dir,
        "run_or_experiment": RandomPolicyTrainer,
        "checkpoint_freq": 25000000,
        "stop":{"training_iteration": 20000},
    },
    "memory": 30 * 10 ** 9,
    "object_store_memory":30 * 10 ** 9
}


debug_config = {
    "memory": 7 * 10 ** 9,
    "object_store_memory": 7 * 10 ** 9,
    "tune_config":{
        "config":{
            "num_workers": 2,
            "num_gpus":0.15,
            "num_gpus_per_worker":0.15,
            "train_batch_size": 256,
            "env_config":{
                "reward_config":{
                    "actor_sync_freq":100,
                    "scoreProxy_options":{"num_cpus":1, "num_gpus":0.3},
                    "scoreProxy_config":{
                        "update_freq": 300,
                        "oracle_config":{"num_threads": 1,},
                        "acquisition_config":{
                            "acq_size": 2,
                            "model_config":{
                                "train_epochs":2,
                                "batch_size":5,
                        }}}}}}}}

rand_001 = {
    "default":config_randpolicy_v1,
            }
