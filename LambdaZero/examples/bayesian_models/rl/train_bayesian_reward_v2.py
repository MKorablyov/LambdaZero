import sys, os, time, socket
import ray
import os.path as osp

from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils import merge_dicts

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils
import LambdaZero.inputs
import torch
import torch_geometric.transforms as T


from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop
from LambdaZero.examples.bayesian_models.rl import config
from LambdaZero.environments.bayesian_reward import BayesianRewardActor
from LambdaZero.environments.persistent_search.persistent_buffer import \
    BlockMolGraphEnv_PersistentBuffer, BlockMolEnvGraph_v1
from LambdaZero.examples.bayesian_models.bayes_tune.functions import get_tau, train_epoch_with_targets, eval_epoch, \
    train_mcdrop_rl, mcdrop_mean_variance
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config
#from LambdaZero.environments import PredDockBayesianReward_v1
from LambdaZero.environments import ProxyReward, ScoreProxy

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "ppo_bayes_reward_008"
config = getattr(config,config_name)
curr_trial = config_name + time.strftime("%Y-%m-%d_%H-%M-%S")


rllib_config = {
    "env": #BlockMolEnvGraph_v1,
          BlockMolEnvGraph_v1, # fixme maybe ray.remote the buffer as well
    "env_config": {
        "random_steps": 4,
        "allow_removal": True,
        "reward": ProxyReward,
        "reward_config": {
        },

    },
    "num_workers": 8,
    "num_gpus_per_worker": 0.25,
    "num_gpus": 1,
    "model": {
        "custom_model": "GraphMolActorCritic_thv1",
        "custom_model_config": {
            "num_blocks": 105, # fixme specify number of blocks only in env?
            "num_hidden": 64
        },
    },
    "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics},  # fixme (report all)
    "framework": "torch",
    "lr": 5e-5,
}

DEFAULT_CONFIG = {
    "rllib_config":rllib_config,
    "summaries_dir": summaries_dir,
    "memory": 30*10**9,
    "object_store_memory": 30*10**9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 2000000},
    #"reward_learner_config":reward_learner_config,
    "use_dock": True,
    "pretrained_model": None # "/home/mjain/scratch/mcdrop_rl/model.pt"
}

config = merge_dicts(DEFAULT_CONFIG, config)

# convenience option to debug on someone's laptop (add yours)
machine = socket.gethostname()
if machine == "Ikarus":
    config["rllib_config"]["num_workers"] = 1
    config["rllib_config"]["num_gpus"] = 0.3
    config["rllib_config"]["memory"] = 25 * 10**9
    config["rllib_config"]["sgd_minibatch_size"] = 4



if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)

    scoreProxy = ScoreProxy.remote(update_freq=20)
    config['rllib_config']['env_config']['reward_config']['scoreProxy'] = scoreProxy

    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])