import socket
from copy import deepcopy
import os.path as osp
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments import PredDockReward_v2
from LambdaZero.environments.persistent_search import BlockMolEnv_PersistentBuffer
from LambdaZero.examples.synthesizability.vanilla_chemprop import DEFAULT_CONFIG as chemprop_cfg

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