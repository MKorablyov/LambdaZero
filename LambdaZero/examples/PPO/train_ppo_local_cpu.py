import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import merge_dicts
import os

import LambdaZero.utils
from LambdaZero.environments import BlockMolEnv_v3, PredDockReward_v2
from LambdaZero.models.torch_models import MolActorCritic_thv1
from LambdaZero.examples.synthesizability.vanilla_chemprop import (
    DEFAULT_CONFIG as chemprop_cfg,
)

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

config = {
    "rllib_config": {
        "log_level": "INFO",
        "env": BlockMolEnv_v3,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v2,
            "reward_config": {
                "synth_cutoff": [0, 5],
                "synth_config": chemprop_cfg,
                "device": "cpu",
                "load_model": os.path.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
                "natm_cutoff": [45, 50],
            },
        },
        "model": {"custom_model": "MolActorCritic_thv1"},
        "use_pytorch": True,
        "num_gpus_per_worker": 0.0,
        "num_gpus": 0.0,
    }
}

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
        "num_workers": 4,
        "num_gpus_per_worker": 0.0,
        "num_gpus": 0.0,
        "model": {"custom_model": "MolActorCritic_thv1"},
        "callbacks": {
            "on_episode_end": LambdaZero.utils.dock_metrics
        },  # fixme (report all)
    },
    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop": {"training_iteration": 3},
}

config = merge_dicts(DEFAULT_CONFIG, config)

if __name__ == "__main__":
    ray.init(memory=config["memory"])
    ModelCatalog.register_custom_model("MolActorCritic_thv1", MolActorCritic_thv1)
    tune.run(
        config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
        local_dir=summaries_dir,
        name="debugging_local_cpu_run",
        checkpoint_freq=config["checkpoint_freq"],
    )
