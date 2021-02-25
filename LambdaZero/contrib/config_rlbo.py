import os.path as osp
import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolEnvGraph_v1
from LambdaZero.environments.reward import PredDockReward_v2
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyReward, ProxyRewardSparse
from LambdaZero.contrib.oracle import DockingOracle
from LambdaZero.contrib.inputs import temp_load_data_v1
from ray.rllib.agents.ppo import PPOTrainer
from LambdaZero.contrib.loggers import log_episode_info

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

from config_model import load_seen_config
from config_acquirer import oracle_config, acquirer_config


proxy_config = {
    "update_freq": 10000,
    "acquirer_config":acquirer_config,
    "oracle": DockingOracle,
    "oracle_config":oracle_config,
    "load_seen": temp_load_data_v1,
    "load_seen_config": load_seen_config,
}

rllib_config = {
    "env": BlockMolEnvGraph_v1, # todo: make ray remote environment
    "env_config": {
        "random_steps": 4,
        "allow_removal": True,
        "reward": ProxyRewardSparse,
        "reward_config": {
            "clip_dockreward":2.5,
            "scoreProxy":ProxyUCB,
            "scoreProxy_config":proxy_config,
            "actor_sync_freq": 500,
        },

    },
    "num_workers": 8,
    "num_gpus_per_worker": 0.05,
    "num_gpus": 0.5,
    "model": {
        "custom_model": "GraphMolActorCritic_thv1",
        "custom_model_config": {
            "num_blocks": 105, # todo specify number of blocks only in env?
            "num_hidden": 64
        },
    },
    "callbacks": {"on_episode_end": log_episode_info},
    "framework": "torch",
    "lr": 5e-5,
    "logger_config":{
        "wandb": {
            "project": "wandb_rlbo",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}}

DEFAULT_CONFIG = {
    "tune_config":{
        "config":rllib_config,
        "local_dir": summaries_dir,
        "run_or_experiment": PPOTrainer,
        "checkpoint_freq": 250,
        "stop":{"training_iteration": 2000},
    },
    "memory": 30 * 10 ** 9,
    "object_store_memory": 30 * 10 ** 9
}

debug_config = {
    "memory": 7 * 10 ** 9,
    "object_store_memory": 7 * 10 ** 9,
    "tune_config":{
        "config":{
            "num_workers": 2,
            "num_gpus":0.3,
            "num_gpus_per_worker":0.05,
            "train_batch_size": 128,
            "sgd_minibatch_size": 4,
            "env_config":{
                "reward_config":{
                    "scoreProxy_config":{
                        "update_freq": 100,
                        "oracle_config":{"num_threads": 2,},
                        "acquirer_config":{
                            "acq_size": 4,
                            "model_config":{
                                "train_epochs":3,
                                "batch_size":10,
                        }}}}}}}}

rlbo_001 = {}
rlbo_002 = {
    "tune_config":{
        "config":{
            "env_config":{
                "reward_config":{
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                    }}}}}}}

rlbo_003 = {
    "tune_config":{
        "config":{
            "env_config":{
                "reward_config":{
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                    }}}}}}}

rlbo_004 = {
    "tune_config":{
        "config":{
            "env_config":{
                "reward_config":{
                    "scoreProxy_config":{
                        "update_freq":1000,
                    }}}}}}

rlbo_005 = {
    "tune_config":{
        "config":{
            "env_config":{
                "random_steps":6,
            }}}}

rlbo_006 = {
    "tune_config":{
        "config":{
            "env_config":{
                "random_steps":6,
                "reward": ProxyReward,
            }}}}

rlbo_007 = { # this maybe needs to run only with EI
    "tune_config":{
        "config":{
            "env_config":{
                "clip_dockreward":None,
                "reward": ProxyReward,
            }}}}
