import socket
from copy import deepcopy
import os
import os.path as osp
from LambdaZero.environments import BlockMolEnv_v3
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v3
from LambdaZero.environments.persistent_search import BlockMolEnv_PersistentBuffer
from LambdaZero.examples.synthesizability.vanilla_chemprop import DEFAULT_CONFIG as chemprop_cfg
from ray.rllib.agents.ppo import PPOTrainer

datasets_dir, programs_dir, summaries_dir = get_external_dirs()
binding_config = deepcopy(chemprop_cfg)
binding_config["predict_config"]["checkpoint_path"] = os.path.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/chemprop/model_0/model.pt")
synth_config = deepcopy(chemprop_cfg)
synth_config["predict_config"]["checkpoint_path"] = os.path.join(datasets_dir, "Synthesizability/MPNN_model/Regression/model_0/model.pt")


ppo001_buf = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_cutoff": [0, 4],
                "ebind_cutoff": [42.5, 109.1], #8.5 std away
                "synth_config": synth_config,
                "binding_config": binding_config,
            }
        }
    },
    "trainer": PPOTrainer,
    "buffer_size": 100_000,
}

ppo001_buf_rnd = {
    # 3.2-3.3
    "rllib_config":{
        "env": BlockMolEnv_PersistentBuffer,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v3,
            "reward_config": {
                "synth_cutoff": [0, 4],
                "ebind_cutoff": [42.5, 109.1], #8.5 std away
                "synth_config": synth_config,
                "binding_config": binding_config,
            }
        },
        "model": {
            "custom_model_config": {
                "rnd_weight": 1
            }
        }
    },
    "trainer": PPOTrainer,
    "buffer_size": 100_000,
}
