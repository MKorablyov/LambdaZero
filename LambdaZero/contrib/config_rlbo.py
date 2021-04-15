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

import LambdaZero.utils
import LambdaZero.contrib.functional
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

from .config_model import load_seen_config
from .config_acquirer import oracle_config, acquirer_config


proxy_config = {
    "update_freq": 1000,
    "acquirer_config":acquirer_config,
    "oracle": DockingOracle,
    "oracle_config":oracle_config,
    "load_seen": temp_load_data_v1,
    "load_seen_config": load_seen_config,
}

trainer_config = { # tune trainable config to be more precise
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

DEFAULT_CONFIG = {
    "tune_config":{
        "config":trainer_config,
        "local_dir": summaries_dir,
        "run_or_experiment": PPOTrainer,
        "checkpoint_freq": 250,
        "stop":{"training_iteration": 2000},
    },
    "memory": 60 * 10 ** 9,
    "object_store_memory": 60 * 10 ** 9
}

debug_config = {
    "memory": 7 * 10 ** 9,
    "object_store_memory": 7 * 10 ** 9,
    "tune_config":{
        "config":{
            "num_workers": 2,
            "num_gpus":0.3,
            "num_gpus_per_worker":0.15,
            "train_batch_size": 256,
            "sgd_minibatch_size": 4,
            "env_config":{
                "reward_config":{
                    "actor_sync_freq":100,
                    "scoreProxy_options":{"num_cpus":1, "num_gpus":0.3},
                    "scoreProxy_config":{
                        "update_freq": 300,
                        "oracle_config":{"num_threads": 1,},
                        "acquirer_config":{
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
                        "acquirer_config":{
                            "model_config":{
                                "device":"cpu"
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


rlbo_007 = {
    "tune_config":{
        "config":{
            "env_config":{
                "reward_config":{
                    "exp_dock":1.5,
                    }}}}}

rlbo_008 = {
    "tune_config":{
        "config":{
            "env_config":{
                "reward_config":{
                    "exp_dock":2,
                    }}}}}

rlbo_009 = {
    "tune_config":{
        "config":{
            "env_config":{
                "reward_config":{
                    "exp_dock":2.5,
                    }}}}}


debug_config_v2 = {
    "tune_config":{
        "config":{
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": True,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":1.0
                        }}}}}}}

debug_config_v3 = {
    "tune_config":{
        "config":{
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":1.0
                        }}}}}}}


debug_config_v4 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": True,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                        }}}}}}}

debug_config_v5 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                        }}}}}}}

debug_config_v6 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": True,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0,
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"],}
                    }}}}}}

debug_config_v7 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

debug_config_v8 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": True,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0,
                            "model_config":{
                                "transform":LambdaZero.utils.Complete(),
                            }
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"],}
                    }}}}}}

debug_config_v9 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0,
                            "model_config":{
                                "transform":LambdaZero.utils.Complete(),
                            }
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"],}
                    }}}}}}


debug_config_v10 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": True,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0,
                            "model_config":{
                                "transform":LambdaZero.utils.Complete(),
                                "mpnn_config":{"drop_weights":False}
                            }
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"],}
                    }}}}}}

debug_config_v11 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0,
                            "model_config":{
                                "transform":LambdaZero.utils.Complete(),
                                "mpnn_config":{"drop_weights":False}
                            }
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"],}
                    }}}}}}


debug_config_v12 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                        "always_discount": True,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0,
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"],}
                    }}}}}}

debug_config_v13 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


# this is a repeat of v7 that worked well
debug_config_v14 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":1,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

# this is a new default
debug_config_v15 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

# this will test with a different pretrain set
debug_config_v16 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,

                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_3k"], }
                    }}}}}}


debug_config_v17 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.3
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

debug_config_v18 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.9
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

debug_config_v19 = {
    # this did not run for the first time
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":2.7
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v20 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":0.9,
                            "model_config": {"mpnn_config": {"drop_weights": False}},
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

debug_config_v21 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":2.7,
                            "model_config": {"mpnn_config": {"drop_weights": False}},
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

debug_config_v22 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": False,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v23 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": False,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 0.9
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v24 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": False,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 2.7
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

debug_config_v25 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa": 4
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v26 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa": 6
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}

debug_config_v27 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": False,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":2.7
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_3k"], }
                    }}}}}}

debug_config_v28 = {
    "tune_config":{
        "config":{
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "env_config":{
                "random_steps":6,
                "reward_config":{
                    "always_discount": True,
                    "scoreProxy_config":{
                        "acquirer_config":{
                            "kappa":2.7
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v29 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": True,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v30 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": True,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 0.9
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v31 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": True,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 2.7
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v32 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
                "max_blocks": 10,
                "max_atoms": 75,
                "max_branches": 40,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": True,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 0.0
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v33 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "max_blocks": 10,
                "max_atoms": 75,
                "max_branches": 40,
                "random_steps": 6,
             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": True,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 0.9
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}


debug_config_v34 = {
    "tune_config": {
        "config": {
            "lr": 5e-5,
            "entropy_coeff": 1e-3,
            "model":{"custom_model_config": {"num_blocks": 464}},

            "env_config": {
                "random_steps": 6,
                "max_blocks": 10,
                "max_atoms": 75,
                "max_branches": 40,

             "molMDP_config": {
                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # more blocks
                 },
                "reward_config": {
                    "always_discount": True,
                    "scoreProxy_config": {
                        "acquirer_config": {
                            "kappa": 2.7
                        },
                        "load_seen_config": {
                            "dataset_split_path": osp.join(datasets_dir,
                            "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                            "file_names": ["Zinc20_docked_neg_randperm_30k"], }
                    }}}}}}