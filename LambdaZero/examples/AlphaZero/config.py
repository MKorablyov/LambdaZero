import socket, time, os.path as osp
from copy import deepcopy
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.utils import merge_dicts

from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer

from LambdaZero.environments import MolMDP, QEDReward, PredDockReward
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4, BlockMolEnv_v5

from .core.alpha_zero_trainer import AlphaZeroTrainer
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

def dock_metrics(info):
    "report custom metrics"
    env_info = list(info["episode"]._agent_to_last_info.values())[0]
    episode = info["episode"]
    episode.custom_metrics["reward"] = env_info["reward"]
    episode.custom_metrics["QED"] = env_info["QED"]
    episode.custom_metrics["discounted_reward"] = env_info["discounted_reward"]



def mol_blocks_v4_config():
    #machine = socket.gethostname()
    obs_config = {"mol_fp_len": 512,
                  "mol_fp_radiis": [3],
                  "stem_fp_len": 64,
                  "stem_fp_radiis": [4, 3, 2]
                  }

    molMDP_config = {
        "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
    }

    reward_config = {
        "soft_stop": True,
        "load_model": osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
        "natm_cutoff": [45, 50],
        "qed_cutoff": [0.2, 0.7],
        "exp": None,
        "delta": False,
        "simulation_cost": 0.00,
        "device": "cuda",
    }

    env_config = {
        "obs_config": obs_config,
        "reward": PredDockReward,
        "molMDP_config": molMDP_config,
        "reward_config": reward_config,
        "num_blocks": 105,
        "max_steps": 7,
        "max_blocks": 7,
        "max_atoms": 50,
        "max_branches": 20,
        "random_blocks": 2,
        "max_simulations": 1,
        "allow_removal": True
    }

    config = {
        "env": BlockMolEnv_v4,
              "env_config": env_config,
              "callbacks": {"on_episode_end": dock_metrics}
              }
    return config



def alphazero_config():
    machine = socket.gethostname()
    if machine == "Ikarus":
        num_workers = 4
        memory = 25 * 10 ** 9
    else:
        num_workers = 19
        memory = 60 * 10 ** 9
    config = {
                 "num_workers": num_workers,
                 "sample_batch_size": 200,
                 "train_batch_size": 4000,
                 "sgd_minibatch_size": 128,
                 "lr": 1e-3,
                 "num_sgd_iter": 1,
                 "mcts_config": {
                     "puct_coefficient": 1.5,
                     "num_simulations": 800,
                     "temperature": 1.0,
                     "dirichlet_epsilon": 0.020,
                     "dirichlet_noise": 0.003,
                     "argmax_tree_policy": False,
                     "add_dirichlet_noise": True,
                 },
                 "ranked_rewards": {
                     "enable": True,
                     "percentile": 75,
                     "buffer_max_length": 1000,
                     # add rewards obtained from random policy to
                     # "warm start" the buffer
                     "initialize_buffer": True,
                     "num_init_rewards": 100,
                 },
                 "model": {
                     "custom_model": "MolActorCritic_thv1",
                     #"custom_options": {"allow_removal": False}
                 },
                 "evaluation_interval": 1,
                 # Number of episodes to run per evaluation period.
                 "evaluation_num_episodes": 1,
                 "num_cpus_per_worker": 1,
                 "num_gpus": 0.4,
                 "num_gpus_per_worker": 0.075,
    }
    checkpoint_freq = 5
    return AlphaZeroTrainer, config, memory, checkpoint_freq


def ppo_config():
    machine = socket.gethostname()
    if machine == "Ikarus":
        num_workers = 5
        memory = 25 * 10 ** 9
    else:
        num_workers = 8
        memory = 60 * 10 ** 9

    config = {
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": num_workers,
        "num_gpus_per_worker": 0.075,
        "num_gpus": 0.4,
        "model": {"custom_model": "MolActorCritic_tfv1"},
        "callbacks": {"on_episode_end": dock_metrics} # fixme !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! other models
    }
    checkpoint_freq = 250
    return PPOTrainer, config, memory, checkpoint_freq


def apex_config():
    machine = socket.gethostname()
    if machine == "Ikarus":
        num_workers = 4
        memory = 25 * 10 ** 9
    else:
        num_workers = 8
        memory = 50 * 10 ** 9

    config = {
        "sample_batch_size": 20,
        "num_envs_per_worker": 1,
        "optimizer": {"num_replay_buffer_shards": 1},
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": num_workers,
        "model": {"custom_model": "MolActorCritic_tfv1"},
        "num_gpus_per_worker": 0.075,
        "num_gpus": 0.4,
        "hiddens": [],
        "dueling": False,
    }
    checkpoint_freq = 250
    return ApexTrainer, config, memory, checkpoint_freq

def get_config(config_name):

    config = deepcopy(eval(config_name))
    #base_env_config = config.pop("base_env_config")()
    trainer, base_trainer_config, memory, checkpoint_freq = config.pop("base_trainer_config")()
    #assert not bool(set(base_env_config.keys()) & set(base_trainer_config.keys())), "default configs overlap"
    #base_config = merge_dicts(base_env_config, base_trainer_config)

    config = merge_dicts(base_trainer_config, config)
    if memory in config.keys(): memory = config.pop("memory")
    return trainer, config, memory, summaries_dir, checkpoint_freq





ppo001 = {
    # 3.2-3.3
    #"base_env_config": mol_blocks_v3_config,
    "base_trainer_config": ppo_config,
    "env": BlockMolEnv_v3,
    "env_config": {
        "allow_removal": True
    }
}
#
# ppo002 = {
#     # 3.1
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_steps": 3,
#         "allow_removal": True
#     }
# }
#
# ppo003 = {
#     # ??
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_steps": 5,
#         "allow_removal": True
#     }
# }
#
# ppo004 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_steps": 3,
#         "max_blocks": 10,
#         "max_atoms": 60,
#         "allow_removal": True,
#         "reward_config": {"natm_cutoff": [50, 60]},
#     }
# }
#
# ppo005 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_steps": 5,
#         "max_blocks": 10,
#         "max_atoms": 60,
#         "allow_removal": True,
#         "reward_config": {"natm_cutoff": [50, 60]},
#     }
# }
#
# ppo006 = {
#     # 3.36
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_steps": 3,
#         "allow_removal": True,
#         "reward_config":{"exp":1.5}
#     }
# }
#
# ppo007 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_steps": 3,
#         "allow_removal": True,
#         "reward_config":{"exp":2.0}
#     }
# }
#
# ppo008 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_steps": 3,
#         "allow_removal": True,
#         "reward_config":{"exp": 2.5}
#     }
# }
#
# ppo009 = {
#     # mean 0.36
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "max_blocks": 10,
#         "max_steps": 14
#     }
# }
#
# ppo010 = {
#     # mean 0.36
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "max_blocks": 10,
#         "max_steps": 7
#     }
# }
#
# ppo011 = {
#     # mean 0.36
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "max_blocks": 10,
#         "max_steps": 10
#     }
# }
#
# ppo012 = {
#     # 3.2-3.3
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "soft_stop": True,
#     }
# }
#
# ppo013 = {
#     # 3.2-3.3
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "soft_stop": False,
#     }
# }
#
# ppo014 = {
#     # 3.1
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "soft_stop": True,
#     }
# }
#
#
# ppo015 = {
#     # 2.67
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_blocks": 5,
#     }
# }
#
# ppo016 = {
#     # 2.66 (slightly slower convergence)
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_blocks": 5,
#         "soft_stop":False,
#     }
# }
#
# ppo017 = {
#     # 2.6 instead of 2.7, 3.05 max instead of 3.12
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_blocks": 5,
#         "max_blocks": 10,
#     }
# }
#
# ppo018 = {
#     # 3.23 instead of 2.68 mean; 3.3 instead of 3.2 final huge improvement !!
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_blocks": 5,
#         "max_steps": 10,
#     }
# }
#
# ppo019 = {
#     # 3.05 mean, 3.08 max
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "random_blocks": 5,
#         "max_steps": 12,
#     }
# }
#
# ppo020 = {
#     "base_env_config": mol_blocks_v4_config,
#     "base_trainer_config": ppo_config,
#     "env": BlockMolEnv_v5,
#     "env_config": {
#         "random_blocks": 5,
#         "max_steps": 10,
#     }
# }
#
# ppo021 = {
#     # 3.2-3.3
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": ppo_config,
#     "env_config": {
#         "allow_removal": True,
#         "num_blocks": 20,
#         "max_branches": 30,
#         "molMDP_config":{ "blocks_file": osp.join(datasets_dir, "fragdb/example_blocks.json")},
#     }
# }
#
#
# apex001 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "max_steps": 7,
#                    "allow_removal": True}
# }
#
# apex002 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "random_steps": 3,
#                    "max_steps": 7,
#                    "allow_removal": True}
# }
#
# apex003 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "random_steps": 5,
#                    "max_steps": 7,
#                    "allow_removal": True}
# }
#
# apex004 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "max_steps": 12,
#                    "allow_removal": True}
# }
#
# apex005 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "random_steps": 3,
#                    "max_steps": 12,
#                    "allow_removal": True}
# }
#
# apex006 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "random_steps": 5,
#                    "max_steps": 12,
#                    "allow_removal": True}
# }
#
# apex007 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "max_steps": 7,
#                    "allow_removal": True,
#                    "reward_config":{"exp": 1.5}
#                    }
# }
#
# apex008 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "random_steps": 3,
#                    "max_steps": 7,
#                    "allow_removal": True,
#                    "reward_config":{"exp": 2.0}
#                    }
# }
#
# apex009 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": apex_config,
#     "buffer_size": 100000, # 16kb/sample # 100000
#     "env_config": {"max_blocks": 9,
#                    "random_blocks": 5,
#                    "max_steps": 7,
#                    "allow_removal": True,
#                    "reward_config": {"exp": 2.5}
#                    }
# }
