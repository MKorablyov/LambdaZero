from LambdaZero.environments import BlockMolEnv_v3 
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v2
from LambdaZero.examples.synthesizability.vanilla_chemprop import DEFAULT_CONFIG as chemprop_cfg

az000 = {
    "rllib_config":{
        "env": BlockMolEnv_v3,
        "env_config":  {
            "obs_config":{
            "mol_fp_len": 512,
            "mol_fp_radiis": [2,3],
            "stem_fp_len": 64,
            "stem_fp_radiis": [2,3,4]
            },
            "max_blocks": 7,
            "reward_config": {"soft_stop":True,"device":"cuda"},
        },
        "sample_batch_size": 128,
        "mcts_config": {
            "num_simulations": 2
        },
    }
}

# az001 = { # killed OOM
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
# }

# az002 = { # max 3.2
#     "mcts_config": {
#         "num_simulations": 30
#     }
# }
#
# az003 = { # max 3.4
#     "mcts_config": {
#         "num_simulations": 90
#     }
# }
#
# az004 = { # killed
#     "mcts_config": {
#         "num_simulations": 270
#     }
# }
#
# az005 = { # killed
#     "mcts_config": {
#         "num_simulations": 1600
#     }
# }
#
# az006 = {
#     "mcts_config": { # max 1.8
#         "dirichlet_epsilon": 0.000020,
#         "dirichlet_noise": 0.00003,
#     }
# }
#
# az007 = {
#     "mcts_config": { # max 2.1
#         "dirichlet_epsilon": 0.00020,
#         "dirichlet_noise": 0.003,
#     }
# }
#
# az008 = { # max 2.4
#     "mcts_config": {
#         "dirichlet_epsilon": 0.006,
#         "dirichlet_noise": 0.0009}
# }
#
# az009 = { # max 2.5
#     "mcts_config": {
#         "dirichlet_epsilon": 0.060,
#         "dirichlet_noise": 0.009,
#     }
# }
#
# az010 = { # max 2.6, climbing
#     "ranked_rewards": {
#         "percentile": 60,
#     }
# }
#
# az011 = { # max 2.2 climbing
#     "ranked_rewards": {
#         "percentile": 55,
#     }
# }
#
# az012 = { # 2.9 climbing
#     "lr":3e-3
# }
#
# az013 = { # 2.5 climbing
#     "lr":3e-4
# }
#
# az014 = { # max 2.9 climbing
#     "num_sgd_iter": 3
# }
#
# az015 = { # max 3.0 climbing
#     "num_sgd_iter": 9
# }
#
# az016 = { #.2.2 climbing
#     "env_config": {"obs_config":
#         {"mol_fp_len": 1024,
#             "stem_fp_len":128}}
# }
#
# az017 = { # 2.4 climbing
#     "env_config": {"obs_config":
#         {"mol_fp_len": 1024,
#             "stem_fp_len": 256}}
# }
#
# az018 = {
#     "env_config": {"reward_config": {"soft_stop": False}}
# }
#
# az019 = {
#     "env_config": {"max_steps": 5}
# }
#
# az020 = {
#     "env_config": {"max_steps": 9}
# }
#
#
# az021 = {
#     "env_config": {"random_steps": 0}
# }
#
# az022 = {
#     "env_config": {"random_steps": 1}
# }
#
#
# az023 = {
#     "ranked_rewards": {
#         "percentile": 60,
#     }
# }
#
# az024 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "ranked_rewards": {
#         "percentile": 55,
#     }
# }
#
# az025 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "lr":3e-3
# }
#
# az026 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "lr":3e-4
# }
#
# az027 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3
# }
#
# az028 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 9
# }
#
# az029 = {
#     # eval max      3.2
#     # eval mean     1.68
#     # max           3.3
#     # mean          1.86
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 27
# }
#
#
# az030 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "ranked_rewards": {
#         "percentile": 60,
#     }
# }
#
# az031 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 9,
#     "ranked_rewards": {
#         "percentile": 60,
#     }
# }
#
# az032 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "lr":3e-4,
#     "num_sgd_iter": 3,
#     "ranked_rewards": {
#         "percentile": 60,
#     }
# }
#
# az033 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "lr":3e-4,
#     "num_sgd_iter": 9,
#     "ranked_rewards": {
#         "percentile": 60,
#     }
# }
#
#
az034 = {
    # eval max      3.1
    # eval mean     2.7
    # eval mean     2.8
    # eval max      3.1
    # max           3.3
    # mean          2.4
    "rllib_config":{
        "env": BlockMolEnv_v3,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v2,
            "reward_config":{
                "synth_cutoff":[0, 5],
                "synth_config": chemprop_cfg
            }

        },
        "num_sgd_iter": 3
    }

}


#
# az035 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 9,
#     "env_config":{"allow_removal":True}
# }
#
# az036 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "ranked_rewards": {
#         "percentile": 60,
#     },
#     "env_config":{"allow_removal":True}
# }
#
# az037 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "ranked_rewards": {
#         "percentile": 55,
#     },
#     "env_config":{"allow_removal":True}
# }
#
#
# az038 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "env_config":{"allow_removal":True, "random_steps": 3}
# }
#
# az039 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 9,
#     "env_config":{"allow_removal":True, "random_steps": 3}
# }
#
# az040 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "ranked_rewards": {
#         "percentile": 60,
#     },
#     "env_config":{"allow_removal":True, "random_steps": 3}
# }
#
# az041 = {
#     # the copy of 34
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "env_config":{"random_steps": 5,
#                   "allow_removal":True}
# }
#
# az042 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "ranked_rewards": {"percentile": 60},
#     "env_config": {
#         "random_steps": 5,
#         "allow_removal": True}
# }
#
# az043 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "ranked_rewards": {"percentile": 85},
#     "env_config": {
#         "random_steps": 5,
#         "allow_removal": True,
#         }
# }
#
# az044 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "env_config":{ "random_steps": 5,
#                   "max_blocks": 10,
#                   "num_steps": 14,
#                   "allow_removal":True}
# }
#
# az045 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "ranked_rewards": {"percentile": 60},
#     "env_config":{ "random_steps": 5,
#                   "max_blocks": 10,
#                   "num_steps": 14,
#                   "allow_removal":True}
# }
#
# az046 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "ranked_rewards": {"percentile": 85},
#     "env_config":{"random_steps": 5,
#                   "max_blocks": 10,
#                   "num_steps": 14,
#                   "allow_removal": True}
# }
#
#
# az047 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "simple_optimizer": False,
#     "buffer_size": 20000,
#     "env_config":{"random_steps": 5,
#                   "allow_removal":True}
# }
#
# az048 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "simple_optimizer": False,
#     "buffer_size": 20000,
#     "ranked_rewards": {"percentile": 60},
#     "env_config": {
#         "random_steps": 5,
#         "allow_removal": True}
# }
#
# az049 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "simple_optimizer": False,
#     "buffer_size": 20000,
#     "ranked_rewards": {"percentile": 85},
#     "env_config": {
#         "random_steps": 5,
#         "allow_removal": True,
#         }
# }
#
# az050 = {
#     "base_env_config": mol_blocks_v3_config,
#     "base_trainer_config": alphazero_config,
#     "num_sgd_iter": 3,
#     "simple_optimizer": False,
#     "buffer_size": 2000,
#     "env_config":{"random_steps": 5,
#                   "allow_removal":True}
# }