import sys, os, time, socket
import ray

from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from LambdaZero.examples.AlphaZero.core.alpha_zero_trainer import AlphaZeroTrainer

from ray.rllib.utils import merge_dicts

from LambdaZero.models.torch_models import MolActorCritic_thv1
from LambdaZero.models.tf_models import MolActorCritic_tfv1
import LambdaZero.utils

from LambdaZero.examples.AlphaZero import config

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "az000"
config = getattr(config,config_name)

_, _, summaries_dir = LambdaZero.utils.get_external_dirs()


DEFAULT_CONFIG = {
    "rllib_config":{
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": 2,
        "sample_batch_size": 200,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "lr": 1e-3,
        "num_sgd_iter": 1,
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 800,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.020,
            "dirichlet_noise": 0.003,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
        },
        "ranked_rewards": {
            "enable": True,
            "percentile": 75,
            "buffer_max_length": 1000,
            # add rewards obtained from random policy to
            # "warm start" the buffer
            "initialize_buffer": True,
            "num_init_rewards": 100,
        },
        "model": {
            "custom_model": "MolActorCritic_thv1",
            "custom_options": {
                "rnd_weight": 0
            }
        },
        "evaluation_interval": 1,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 1,
        "num_cpus_per_worker": 1,
        "num_gpus": 0.4,
        "num_gpus_per_worker": 0.075,
        "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics},
    },
    "summaries_dir": summaries_dir,
    "memory": 100 * 10 ** 9,
    "trainer": AlphaZeroTrainer,
    "checkpoint_freq": 5,
    "stop":{"training_iteration": 2000000},
}

config = merge_dicts(DEFAULT_CONFIG, config)

# convenience option to debug on maksym's personal laptop
machine = socket.gethostname()
if machine == "Ikarus":
    config["rllib_config"]["num_workers"] = 4
    config["rllib_config"]["memory"] = 25 * 10**9



if __name__ == "__main__":
    ray.init(memory=config["memory"])
    time.sleep(60)
    ModelCatalog.register_custom_model("MolActorCritic_thv1", MolActorCritic_thv1)
    #ModelCatalog.register_custom_model("MolActorCritic_tfv1", MolActorCritic_tfv1)
    time.sleep(60)
    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])