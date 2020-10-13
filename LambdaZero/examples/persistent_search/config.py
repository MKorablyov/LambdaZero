import socket
from copy import deepcopy
import os.path as osp
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v2
from LambdaZero.environments.persistent_search import BlockMolEnv_PersistentBuffer, BlockMolGraphEnv_PersistentBuffer
from LambdaZero.examples.synthesizability.vanilla_chemprop import DEFAULT_CONFIG as chemprop_cfg
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config
from ray.rllib.agents.ppo import PPOTrainer


datasets_dir, programs_dir, summaries_dir = get_external_dirs()



ppo001_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v2,
            "reward_config":{
                "synth_cutoff":[0, 5],
                "synth_config": chemprop_cfg}
        }
    },
    "trainer": PPOTrainer,
    "buffer_size": 100_000,
}

ppo001_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo002_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.9
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo003_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.8
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo004_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.6
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo005_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.5
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo006_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 1.0
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo007_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.6,
            "random_start_prob": 0.5
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo007_graph_buf_1 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.6,
            "random_start_prob": 0.25
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo007_graph_buf_2 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.6,
            "random_start_prob": 0.75
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo009_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.7,
            "random_start_prob": 0.5
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo009_graph_buf_1 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.7,
            "random_start_prob": 0.25
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo009_graph_buf_2 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.7,
            "random_start_prob": 0.75
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo0010_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.5,
            "random_start_prob": 0.5
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo0010_graph_buf_1 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.5,
            "random_start_prob": 0.25
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo0010_graph_buf_2 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            "threshold": 0.5,
            "random_start_prob": 0.75
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}

ppo008_graph_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            },
            # "threshold": 0.6,
            # "random_start_prob": 0.5
            "use_similarity": False
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{
                "num_hidden": 64, # does a **kw to __init__
            }
        },
        "lr": 5e-5,
        #"entropy_coeff": 1e-5,
        "framework": "torch",
    },
    "buffer_size": 500_000,
}