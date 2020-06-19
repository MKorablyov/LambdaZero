import time

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import merge_dicts
from ray.tune import Stopper

import LambdaZero.utils
from LambdaZero.environments import BlockMolEnvGraph_v1, PredDockReward_v2
from LambdaZero.examples.synthesizability.vanilla_chemprop import (
    DEFAULT_CONFIG as chemprop_cfg,
)
from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1


class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        self._deadline = 10

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self._deadline


_, _, summaries_dir = LambdaZero.utils.get_external_dirs()

ppo_graph_001 = {
    "rllib_config": {
        "env": BlockMolEnvGraph_v1,
        "env_config": {
            "allow_removal": True,
            "reward": PredDockReward_v2,
            "reward_config": {"synth_cutoff": [0, 5],
                              "device": "cpu",
                              "synth_config": chemprop_cfg},
        },
        "lr": 1e-4,
        "model": {"custom_model": "GraphMolActorCritic_thv1"},
#        "framework": "torch",
        "use_pytorch": True,  # I seem to be using an older version of ray...
    },
    "checkpoint_freq": 10,
}

DEFAULT_CONFIG = {
    "rllib_config": {
        "tf_session_args": {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        },
        "local_tf_session_args": {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        },
        "num_workers": 1,
        "num_gpus_per_worker": 0.0,
        "num_gpus": 0.0,
        "model": {"custom_model": "MolActorCritic_tfv1"},
        "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics},
    },
    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop": {"training_iteration": 1},
}

config = merge_dicts(DEFAULT_CONFIG, ppo_graph_001)


if __name__ == "__main__":
    #profiler = Profiler()
    #profiler.start()

    ray.init(local_mode=True)
    #ray.init()
    ModelCatalog.register_custom_model(
        "GraphMolActorCritic_thv1", GraphMolActorCritic_thv1
    )
    tune.run(
        config["trainer"],
        stop=TimeStopper(),
        max_failures=0,
        config=config["rllib_config"],
        local_dir=summaries_dir,
        name="debugging_graph_ppo",
        checkpoint_freq=config["checkpoint_freq"],
    )

