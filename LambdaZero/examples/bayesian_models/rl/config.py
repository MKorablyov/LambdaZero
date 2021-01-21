import socket
from copy import deepcopy
import os
import os.path as osp
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4, BlockMolEnvGraph_v1
from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolGraphEnv_PersistentBuffer
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v2, PredDockReward_v3, PredDockBayesianReward_v1
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

# fixme
#binding_model = osp.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/vanilla_mpnn/model.pth")
binding_model = osp.join(datasets_dir, "brutal_dock/seh/trained_weights/vanilla_mpnn/model.pth")
# binding_config["predict_config"]["checkpoint_path"] = \
#     os.path.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/chemprop/model_0/model.pt")

# synth_config = deepcopy(chemprop_cfg)
# synth_config["predict_config"]["checkpoint_path"] = \
#     os.path.join(datasets_dir, "Synthesizability/MPNN_model/Regression/model_0/model.pt")

ppo_bayes_reward_000 = {
    "rllib_config": {
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config": {"num_hidden": 64} # does a **kw to __init__
        },
        # "lr": 1e-4,
        "framework": "torch",
    },
}

ppo_bayes_reward_000_1 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
}

ppo_bayes_reward_001 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 1e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },

}

ppo_bayes_reward_002 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "aq_size0": 10000,
    }
}


ppo_bayes_reward_003 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        'num_mol_retrain': 5000,
        "sync_freq": 650,
    }
}

ppo_bayes_reward_004 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        'num_mol_retrain': 2000,
        "sync_freq": 275,
    }
}


ppo_bayes_reward_005 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 15,
            "max_blocks": 10,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
}

ppo_bayes_reward_006 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
}

ppo_bayes_reward_008 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        'num_mol_retrain': 1000,
    },
    "use_dock": True
}

ppo_bayes_reward_008_1 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "aq_size": 64,
        'num_mol_retrain': 2000,
    },
    "use_dock": True
}

ppo_bayes_reward_008_2 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "aq_size": 128,
        'num_mol_retrain': 3000,
    },
    "use_dock": True
}

ppo_bayes_reward_008_3 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "aq_size": 4,
        'num_mol_retrain': 100,
    },
    "use_dock": True
}

ppo_bayes_reward_008_4 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "molMDP_config": {
                "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # larger building block action space
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        'num_mol_retrain': 1000,
    },
    "use_dock": True
}

ppo_bayes_reward_007 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        'num_mol_retrain': 1000,
    },
    "use_dock": True
}

ppo_bayes_reward_007_2 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "molMDP_config": {
                "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"), # larger building block action space
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        'num_mol_retrain': 1000,
    },
    "use_dock": True
}

ppo_bayes_reward_009 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "random_start_prob": 0.5,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "use_dock": True
    # "reward_learner_config": {
    #     "train_epochs": 100
    # },
}

ppo_bayes_reward_009_1 = {
    "rllib_config":{
        "env": BlockMolGraphEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "random_start_prob": 0.25,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "use_dock": True
    # "reward_learner_config": {
    #     "train_epochs": 100
    # },
}

ppo_bayes_reward_010 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 0
    },
    "use_dock": True
}

ppo_bayes_reward_010_1 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 0
    },
}

ppo_bayes_reward_011 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 0.3
    },
    "use_dock": True
}

ppo_bayes_reward_011_1 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 0.3
    }
}

ppo_bayes_reward_012 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_012_1 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
}

ppo_bayes_reward_013 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 2
    },
    "use_dock": True
}

ppo_bayes_reward_014 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 10
    },
    "use_dock": True
}

ppo_bayes_reward_015 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 20
    },
    "use_dock": True
}

ppo_bayes_reward_016 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 5
    },
    "use_dock": True
}

ppo_bayes_reward_017 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 50
    },
    "use_dock": True
}

ppo_bayes_reward_018 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "random_steps": 0,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_019 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
                "delta": True,
                "dense_rewards": True
            },
            "random_steps": 0,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_020 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_021 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 10
    },
    "use_dock": True
}

ppo_bayes_reward_022 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 7,
            "random_steps": 2,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_023 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 7,
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_024 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 7,
            "random_steps": 6,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_025 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 10,
            "random_steps": 2,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}
ppo_bayes_reward_026 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 10,
            "random_steps": 4,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}

ppo_bayes_reward_027 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 10,
            "random_steps": 6,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1
    },
    "use_dock": True
}


ppo_bayes_reward_028 = { # drop weights and last layer
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 10,
            "random_steps": 6,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1,
        'regressor_config' : {
            "model_config": {"drop_data":False, "drop_weights":True, "drop_last":True, "drop_prob":0.1},
        }
    },
    "use_dock": True
}

ppo_bayes_reward_029 = { # incr dropout rate
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 10,
            "random_steps": 6,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1,
        'regressor_config' : {
            "model_config": {"drop_data":False, "drop_weights":True, "drop_last":True, "drop_prob":0.2},
        }
    },
    "use_dock": True
}

ppo_bayes_reward_030 = { # drop only weights (not last layer)
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "binding_model": binding_model,
            },
            "max_steps": 10,
            "random_steps": 6,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_model_config":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        "entropy_coeff": 1e-3,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "reward_learner_config": {
        "num_mol_retrain": 1000,
        "kappa": 1,
        'regressor_config' : {
            "model_config": {"drop_data":False, "drop_weights":True, "drop_last":False, "drop_prob":0.1},
        }
    },
    "use_dock": True
}
