import sys, os, time, socket
import ray

from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils import merge_dicts

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils
import torch

from LambdaZero.environments import block_mol_v3
from LambdaZero.examples.PPO_RND import config
from LambdaZero.environments.reward import BayesianRewardActor

from LambdaZero.examples.bayesian_models.bayes_tune import config
from LambdaZero.examples.bayesian_models.bayes_tune.functions import get_tau, train_epoch_with_targets, eval_epoch, \
    train_mcdrop, mcdrop_mean_variance

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "ppo001_buf"
config = getattr(config,config_name)

DEFAULT_CONFIG = {
    "rllib_config":{
        "regressor_config": {
            "lambda": 1e-8,
            "T": 20,
            "lengthscale": 1e-2,
            "uncertainty_eval_freq":15,
            "train_iterations":150,
            "model": LambdaZero.models.MPNNetDrop,
            "model_config": {"drop_data":False, "drop_weights":False, "drop_last":True, "drop_prob":0.1},
            "optimizer": torch.optim.Adam,
            "optimizer_config": {
                "lr": 0.001
            },
            "train_epoch": train_epoch_with_targets,
            "eval_epoch": eval_epoch,
            "train": train_mcdrop,
            "get_mean_variance": mcdrop_mean_variance,
        },
        "reward_learner_config": {
            "aq_size0": 200,
            "aq_size": 50,
            "kappa": 0.2,
            "epsilon": 0.0,
            "minimize_objective":True,
            "b_size": 32,
        },
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": 11,
        "num_envs_per_worker": 2,
        "num_gpus_per_worker": 0.075,
        "num_gpus": 2,
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
        },
        "env_config": {
            "threshold": 0.7
        },
        "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics}, # fixme (report all)
        "framework": "torch",
        },
    "summaries_dir": summaries_dir,
    "memory": 70 * 10 ** 9,
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
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)
    
    reward_learner = BayesianRewardActor.remote(config['regressor_config'])

    searchbuf = ray.remote(num_gpus=1)(PersistentSearchBuffer).remote(
        {'blocks_file': block_mol_v3.DEFAULT_CONFIG['molMDP_config']['blocks_file'],
         'max_size': config['buffer_size'],
         'threshold': config["rllib_config"]['env_config']['threshold']})
    config['rllib_config']['env_config']['searchbuf'] = searchbuf

    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])