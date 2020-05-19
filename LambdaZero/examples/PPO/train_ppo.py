import sys, os, time, socket
import ray

from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer

from ray.rllib.utils import merge_dicts

from LambdaZero.models.torch_models import MolActorCritic_thv1
from LambdaZero.models.tf_models import MolActorCritic_tfv1
import LambdaZero.utils

from LambdaZero.examples.PPO import config

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "ppo001" # "apex001"
config = getattr(config,config_name)

_, _, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = {
    "rllib_config":{
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": 7,
        "num_gpus_per_worker": 0.075,
        "num_gpus": 0.4,
        "model": {"custom_model": "MolActorCritic_tfv1"},
        "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics}
        },
    "summaries_dir": summaries_dir,
    "memory": 60 * 10 ** 9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 2000000},
}

config = merge_dicts(DEFAULT_CONFIG, config)

# convenience option to debug on maksym's personal laptop
machine = socket.gethostname()
if machine == "Ikarus":
    config["rllib_config"]["num_workers"] = 5
    config["rllib_config"]["memory"] = 25 * 10**9

if __name__ == "__main__":
    ray.init(memory=config["memory"])
    ModelCatalog.register_custom_model("MolActorCritic_thv1", MolActorCritic_thv1)
    ModelCatalog.register_custom_model("MolActorCritic_tfv1", MolActorCritic_tfv1)
    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])