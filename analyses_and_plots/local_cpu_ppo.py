"""
The goal of this script is to run an rl algorithm on a local machine without a GPU. This is useful
for quick debugging.
"""

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import merge_dicts
import os

import LambdaZero.utils
from LambdaZero.environments import PredDockReward_v2, BlockMolEnv_v3
from LambdaZero.examples.synthesizability.vanilla_chemprop import (
    DEFAULT_CONFIG as chemprop_cfg,
)
from LambdaZero.models import MolActorCritic_thv1

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = {
    "rllib_config": {
        "tf_session_args": {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        },
        "local_tf_session_args": {
            "intra_op_parallelism_threads": 4,
            "inter_op_parallelism_threads": 4,
        },
        "num_workers": 7,
        "num_gpus_per_worker": 0.075,
        "num_gpus": 0.4,
        "model": {"custom_model": "MolActorCritic_tfv1"},
        "callbacks": {
            "on_episode_end": LambdaZero.utils.dock_metrics
        },  # fixme (report all)
    },
    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop": {"training_iteration": 2000000},
}

ppo024 = {
    "rllib_config": {
        "env": BlockMolEnv_v3,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v2,
            "molMDP_config": {
                "blocks_file": os.path.join(datasets_dir, "fragdb/pdb_blocks_105.json"),
            },
            "reward_config": {
                "synth_cutoff": [0, 5],
                "synth_config": chemprop_cfg,
                "device": "cpu",
            },
        },
        "model": {"custom_model": "MolActorCritic_thv1"},
        "use_pytorch": True,
        "num_gpus_per_worker": 0.0,
        "num_gpus": 0.0,
    }
}

config = merge_dicts(DEFAULT_CONFIG, ppo024)


if __name__ == "__main__":
    ray.init(memory=config["memory"])
    ModelCatalog.register_custom_model("MolActorCritic_thv1", MolActorCritic_thv1)

    tune.run(
        config["trainer"],
        stop={"training_iteration": 3},
        max_failures=0,
        config=config["rllib_config"],
        local_dir=summaries_dir,
        name="debugging_ppo",
        checkpoint_freq=config["checkpoint_freq"],
    )
