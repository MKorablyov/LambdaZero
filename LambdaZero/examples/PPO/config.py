import socket
from copy import deepcopy
import os
import os.path as osp
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4, BlockMolEnvGraph_v1
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v2, PredDockReward_v3
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

ppo001 = {
    "rllib_config":{
        "env": BlockMolEnv_v3,
        "env_config": {
            "allow_removal": True,
        }
    }
}

ppo002 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolEnv_v4,
        "env_config": {
            "allow_removal": True,
        }
    }
}

ppo_mpro_v001 = {
    "rllib_config":{
        "env": BlockMolEnv_v3,
        },
}

ppo_mpro_v002 = {
    "rllib_config":{
        "env": BlockMolEnv_v4,
        },
}





# ppo024 = {
#      "rllib_config":{
#          "env": BlockMolEnv_v3,
#          "env_config": {
#              "allow_removal": True,
#              "reward": PredDockReward_v3,
#              "reward_config": {
#                  "synth_config": synth_config,
#                  "binding_config": binding_config,
#              }
#
#          },
#      }
# }



# ppo022 = {
#     # ???
#     "rllib_config":{
#         "env": BlockMolEnv_v3,
#         "env_config": {
#             "allow_removal": True,
#             "reward": PredDockReward_v3,
#             "reward_config": {
#                 "synth_cutoff": [0, 4],
#                 "ebind_cutoff": [42.5, 109.1], #8.5 std away
#                 "synth_config": synth_config,
#                 "binding_config": binding_config,
#             }
#
#         },
#     }
# }
#
# ppo023 = {
#     # 3.2-3.3
#     "rllib_config":{
#         "env": BlockMolEnv_v3,
#         "env_config": {
#             "molMDP_config": {
#                 "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_210.json"),
#             },
#             "allow_removal": True,
#         }
#     }
# }


#
# ppo_graph_001 = {
#     "rllib_config":{
#         "env": BlockMolEnvGraph_v1,
#         "env_config": {
#             "allow_removal": True,
#             "reward": PredDockReward_v3,
#             "reward_config": {
#                 "synth_cutoff": [0, 4],
#                 "ebind_cutoff": [42.5, 109.1], #8.5 std away
#                 "synth_config": synth_config,
#                 "binding_config": binding_config,
#             }
#         },
#         "model": {"custom_model": "GraphMolActorCritic_thv1",
#                   "custom_options":{"num_hidden": 64} # does a **kw to __init__
#         },
#         "lr": 5e-5,
#         #"entropy_coeff": 1e-5,
#         "use_pytorch": True,
#     },
#     "checkpoint_freq": 25,
# }

# "reward_config": {
#     "soft_stop": True,
#     "load_model": osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
#     "natm_cutoff": [45, 50],
#     "qed_cutoff": [0.2, 0.7],
#     "exp": None,
#     "delta": False,
#     "simulation_cost": 0.0,
#     "device": "cuda",
# },




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
