import socket
from copy import deepcopy
import os
import os.path as osp
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4, BlockMolEnvGraph_v1
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v2, PredDockReward_v3, PredDockBayesianReward_v1
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

# fixme
# binding_config = deepcopy(chemprop_cfg)
# binding_config["predict_config"]["checkpoint_path"] = \
#     os.path.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/chemprop/model_0/model.pt")

# synth_config = deepcopy(chemprop_cfg)
# synth_config["predict_config"]["checkpoint_path"] = \
#     os.path.join(datasets_dir, "Synthesizability/MPNN_model/Regression/model_0/model.pt")


ppo000 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolEnv_v3,
        "env_config": {
            "allow_removal": True,
        }
    }
}

ppo001 = {
    "rllib_config":{
        "env": BlockMolEnv_v3,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v2,
            "reward_config": {
                "synth_cutoff": [0, 5],
                # "synth_config": chemprop_cfg,
            }
        }
    }
}

ppo002 = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "dockscore_config": binding_config,
                "synth_config": synth_config
            }
        }
    }
}

ppo_mpro_v001 = {
    "rllib_config":{
        "env": BlockMolEnv_v3,
        },
}

ppo_mpro_v002 = {
    "rllib_config": {
        "env": BlockMolEnv_v4,
        },
}

ppo_graph_000 = {
    "rllib_config":{
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockBayesianReward_v1,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
            }
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
            "custom_options":{"num_hidden": 64} # does a **kw to __init__
        },
        "lr": 5e-5,
        # "entropy_coeff": 1e-4,
        # "entropy_coeff_schedule": [(0, 1e-4), (10000, 5e-5), (100000, 1e-5), (1000000, 1e-6)],
        "framework": "torch",
    },
    "checkpoint_freq": 25,
}
