import os.path as osp
import LambdaZero.utils
from LambdaZero.examples.baselines.boltzmann.boltzmann_trainer import BoltzmannTrainer, on_episode_start
from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolEnvGraph_v1
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyReward, ProxyRewardSparse
from LambdaZero.contrib.oracle import DockingOracle
from LambdaZero.contrib.inputs import temp_load_data_v1
from LambdaZero.contrib.loggers import log_episode_info

import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

from LambdaZero.contrib.config_model import load_seen_config
from LambdaZero.contrib.config_acquirer import oracle_config, acquirer_config


proxy_config = {
    "update_freq": 1000,
    "acquirer_config":acquirer_config,
    "oracle": DockingOracle,
    "oracle_config":oracle_config,
    "load_seen": temp_load_data_v1,
    "load_seen_config": load_seen_config,
}
trainer_config = { # tune trainable config to be more precise
    "env": BlockMolEnvGraph_v1,
    "env_config": {
        "temperature": 0.1,
        "random_steps": 20,
        "allow_removal": True,
        "reward": ProxyRewardSparse,
        "reward_config": {
            "synth_options":{"num_gpus":0.05},
            "qed_cutoff": [0.2, 0.5],
            "synth_cutoff":[0, 4],
            "scoreProxy":ProxyUCB,
            "scoreProxy_config":proxy_config,
            "scoreProxy_options":{"num_cpus":2, "num_gpus":1.0},
            "actor_sync_freq": 150,
        },

    },
    "num_workers": 8,
    "num_gpus_per_worker": 0.15,
    "num_gpus": 1.0,
    "model": {
        "custom_model": "GraphMolActorCritic_thv1",
        "custom_model_config": {
            "num_blocks": 105,
            "num_hidden": 64
        },
    },
    "callbacks": {"on_episode_start": on_episode_start,
                  "on_episode_end": log_episode_info},
    "framework": "torch",
    "lr": 5e-5,
    "logger_config": {
        "wandb": {
            "project": "boltzmann",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}
}


DEFAULT_CONFIG = {
    "tune_config":{
        "config":trainer_config,
        "local_dir": summaries_dir,
        "run_or_experiment": BoltzmannTrainer,
        "checkpoint_freq": 250,
        "stop":{"training_iteration": 20000},
    },
    "memory": 3 * 10 ** 9,
    "object_store_memory": 3 * 10 ** 9
}

boltzmann_config_001 = {}