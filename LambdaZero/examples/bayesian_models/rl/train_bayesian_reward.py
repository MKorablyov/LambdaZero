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

from LambdaZero.environments import block_mol_v3
from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop
from LambdaZero.examples.bayesian_models.rl import config
from LambdaZero.examples.bayesian_models.rl.random_search import RandomSearchTrainer
from LambdaZero.environments.reward import BayesianRewardActor

# from LambdaZero.examples.bayesian_models.bayes_tune import config
from LambdaZero.examples.bayesian_models.bayes_tune.functions import get_tau, train_epoch_with_targets, eval_epoch, \
    train_mcdrop_rl, mcdrop_mean_variance

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


data_config = {
    "target": "gridscore",
    # "dataset_creator": LambdaZero.inputs.dataset_creator_v1,
    "dataset_split_path": osp.join(datasets_dir,
                                #    "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
    "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
    "dataset": LambdaZero.inputs.BrutalDock,
    "dataset_config": {
        "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
        "props": ["gridscore", "smi"],
        "transform": T.Compose([LambdaZero.utils.Complete()]),
        "file_names": 
        # ["Zinc15_2k"],
        ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
    },
    "b_size": 40,
    "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057])
}


if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "ppo_bayes_reward_000"
config = getattr(config,config_name)

DEFAULT_CONFIG = {
    "rllib_config":{
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": 7,
        "num_gpus_per_worker": 0.075,
        "num_gpus": 1,
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
        },
        "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics}, # fixme (report all)
        "framework": "torch",

        # "rollout_fragment_length": 1500,
        # "train_batch_size": 200,
        # "num_workers": 4,
        # # "num_gpus_per_worker": 0.075,
        # # "num_gpus": 0,
        # # "model": {
        # #     "custom_model": "GraphMolActorCritic_thv1",
        # # },
        # "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics}, # fixme (report all)
        # "framework": "torch",
        },
    "summaries_dir": summaries_dir,
    "memory": 30 * 10 ** 9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 2000000},
    "regressor_config": {
        "lambda": 1e-8,
        "data": dict(data_config, **{"dataset_creator":None}),
        "T": 20,
        "lengthscale": 1e-2,
        "uncertainty_eval_freq":15,
        "train_iterations": 120,
        "model": LambdaZero.models.MPNNetDrop,
        "model_config": {"drop_data":False, "drop_weights":False, "drop_last":True, "drop_prob":0.1},
        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },
        "train_epoch": train_epoch_with_targets,
        "eval_epoch": eval_epoch,
        "train": train_mcdrop_rl,
        "get_mean_variance": mcdrop_mean_variance,
    },
    "reward_learner_config": {
        "aq_size0": 400,
        "data": dict(data_config, **{"dataset_creator":None}),
        "aq_size": 50,
        "kappa": 0.2,
        "epsilon": 0.0,
        "minimize_objective":True,
        "b_size": 40,
        'num_mol_retrain': 10000
    },
}

config = merge_dicts(DEFAULT_CONFIG, config)

# convenience option to debug on maksym's personal laptop
machine = socket.gethostname()
if machine == "Ikarus":
    config["rllib_config"]["num_workers"] = 5
    config["rllib_config"]["memory"] = 25 * 10**9

if __name__ == "__main__":
    ray.init(memory=config["memory"], num_gpus=4, num_cpus=12)
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)
    
    reward_learner = BayesianRewardActor.options(max_concurrency=4).remote(config['reward_learner_config'], MCDrop,
                                                config['regressor_config'], 
                                                config["rllib_config"]['env_config']['reward_config']['dockscore_config'],
                                                "cuda")

    config['rllib_config']['env_config']['reward_config']['reward_learner'] = reward_learner
    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])