import os.path as osp
from ray.rllib.agents.ppo import PPOTrainer
import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolEnvGraph_v1
from LambdaZero.contrib.environments import BlockMolGraph_v2
from LambdaZero.environments.reward import PredDockReward_v2
from LambdaZero.contrib.proxy import ProxyUCB, config_ProxyUCB_v1
from LambdaZero.contrib.reward import ProxyReward, ProxyRewardSparse, ProxyRewardSparse_v2
from LambdaZero.contrib.oracle import DockingOracle
from LambdaZero.contrib.data import temp_load_data
from LambdaZero.contrib.loggers import log_episode_info
import LambdaZero.contrib.functional

import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

config_rlbo_run_v1 = { # tune trainable config to be more precise
    "env": BlockMolEnvGraph_v1, # todo: make ray remote environment
    "env_config": {
        "random_steps": 4,
        "allow_removal": True,
        "reward": ProxyRewardSparse,
        "reward_config": {
            "synth_options":{"num_gpus":0.05},
            "qed_cutoff": [0.2, 0.5],
            "synth_cutoff":[0, 4],
            "scoreProxy":ProxyUCB,
            "scoreProxy_config": config_ProxyUCB_v1,
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
            "num_blocks": 105, # todo specify number of blocks only in env?
            "num_hidden": 64
        },
    },
    "callbacks": {"on_episode_end": log_episode_info},
    "framework": "torch",
    "lr": 5e-5,
    "logger_config":{
        "wandb": {
            "project": "rlbo4",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}}


config_rlbo_run_v2 = { # tune trainable config to be more precise
    "env": BlockMolGraph_v2,
    "env_config": {
        "reward": ProxyRewardSparse_v2,
        "reward_config": {
            "scoreProxy": ProxyUCB,
            "scoreProxy_config": config_ProxyUCB_v1,
            "scoreProxy_options": {"num_cpus": 2, "num_gpus": 1.0},
        }
    },

    "num_workers": 8,
    "num_gpus_per_worker": 0.15,
    "num_gpus": 1.0,
    "model": {
        "custom_model": "GraphMolActorCritic_thv1",
        "custom_model_config": {
            "num_blocks": 464,
            "num_hidden": 64 # todo - may not be right for 464 blocks
        },
    },
    "callbacks": {"on_episode_end": log_episode_info},
    "framework": "torch",
    "lr": 5e-5,
    "logger_config":{
        "wandb": {
            "project": "rlbo4",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}}


config_rlbo_v1 = {
    "tune_config":{
        "config": config_rlbo_run_v1,
        "local_dir": summaries_dir,
        "run_or_experiment": PPOTrainer,
        "checkpoint_freq": 250,
        "stop":{"training_iteration": 2000},
    },
    "memory": 60 * 10 ** 9,
    "object_store_memory": 60 * 10 ** 9
}


config_rlbo_v2 = {
    "tune_config":{
        "config": config_rlbo_run_v2,
        "local_dir": summaries_dir,
        "run_or_experiment": PPOTrainer,
        "checkpoint_freq": 250,
        "stop":{"training_iteration": 2000},
    },
    "memory": 60 * 10 ** 9,
    "object_store_memory": 60 * 10 ** 9
}


config_debug = {
    "memory": 7 * 10 ** 9,
    "object_store_memory": 7 * 10 ** 9,
    "tune_config":{
        "config":{
            "num_workers": 2,
            "num_gpus":0.3,
            "num_gpus_per_worker":0.15,
            "train_batch_size": 250,
            "sgd_minibatch_size": 4,
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

config_cpu = {
    "tune_config":{
        "config":{
            "num_gpus":0,
            "num_gpus_per_worker":0,
            "env_config":{
                "reward_config":{
                    "synth_options":{"num_gpus":0},
                    "scoreProxy_options":{"num_cpus":1, "num_gpus":0},
                    "scoreProxy_config":{
                        "acquisition_config":{
                            "model_config":{
                                "device":"cpu"
                        }}}}}}}}

rlbo4_001 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_002 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 0.3
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}


rlbo4_003 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 0.
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_004 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisiton_config": {
                            "kappa": 3.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_005 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 5,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}


rlbo4_006 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k_3k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}


rlbo4_007 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 0.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k_3k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}



rlbo4_008 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 5e-4,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_009 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 2e-4,
            "model":{"custom_model_config": {"num_blocks": 464}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_010 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config": {
                "random_steps": 4,
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_011 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config": {
                "random_steps": 4,
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 5.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_012 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config": {
                "random_steps": 4,
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 10.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}




rlbo4_013 = {
    "default":config_rlbo_v1,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 2e-4,
            "model":{"custom_model_config": {"num_blocks": 360}},
            "env_config": {
                "random_steps": 4,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55_manFix2.json"), # more blocks
                 },
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                        "load_seen_config": {
                            "mean":None, "std":None, "act_y":None,
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
                            "file_names": ["random_molecule_proxy_20k"], }
                    }}}}}}

rlbo4_014 = {
    "default":config_rlbo_v2,
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config": {
                "reward_config": {
                    "scoreProxy_config": {
                        "acquisition_config": {
                            "kappa": 1.0
                        },
                    }}}}}}


