import socket
from copy import deepcopy
import os
import os.path as osp
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4, BlockMolEnvGraph_v1
from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolGraphEnv_PersistentBuffer
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v2, PredDockReward_v3, PredDockBayesianReward_v1
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config
from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

# fixme
binding_model = osp.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/vanilla_mpnn/model.pth")

az_bayes_reward_000 = {
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
            "custom_options":{"num_hidden": 64} # does a **kw to __init__
        },
        "mcts_config": {
            "num_simulations": 50,
            "policy_optimization": True,
        },
    },
}

az_bayes_reward_001 = {
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
            "custom_options":{"num_hidden": 64} # does a **kw to __init__
        },
        "mcts_config": {
            "num_simulations": 50,
            "policy_optimization": True,
        },
    },
}

az_bayes_reward_001_1 = {
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
            "custom_options":{"num_hidden": 64} # does a **kw to __init__
        },
        "mcts_config": {
            "num_simulations": 50,
            "policy_optimization": True,
        },
    },
    "use_dock": True
}

az_bayes_reward_002 = {
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
            "custom_options":{"num_hidden": 64} # does a **kw to __init__
        },
        "mcts_config": {
            "num_simulations": 25,
            "policy_optimization": True,
        },
    },
}

az_bayes_reward_002_1 = {
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
            "custom_options":{"num_hidden": 64} # does a **kw to __init__
        },
        "mcts_config": {
            "num_simulations": 25,
            "policy_optimization": True,
        },
    },
    "use_dock": False
}