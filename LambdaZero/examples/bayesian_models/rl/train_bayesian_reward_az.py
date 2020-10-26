import sys, os, time, socket
import ray
import os.path as osp

from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from LambdaZero.examples.AlphaZero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.callbacks import DefaultCallbacks

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils

import LambdaZero.inputs
import torch
import torch_geometric.transforms as T

# from LambdaZero.environments import block_mol_v3
from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop
from LambdaZero.examples.bayesian_models.rl import config_az as config
from LambdaZero.examples.bayesian_models.rl.random_search import RandomSearchTrainer
from LambdaZero.environments.reward import BayesianRewardActor

# from LambdaZero.examples.bayesian_models.bayes_tune import config
from LambdaZero.examples.bayesian_models.bayes_tune.functions import get_tau, train_epoch_with_targets, eval_epoch, \
    train_mcdrop_rl, mcdrop_mean_variance
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

class AZCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode):
        # print("episode {} started".format(episode.episode_id))
        episode.user_data["initial_state"] = base_env.get_unwrapped()[0].get_state()

    def on_episode_end(self, worker, base_env, policies, episode):
        env_info = list(episode._agent_to_last_info.values())[0]

        for key, value in env_info["log_vals"].items():
            episode.custom_metrics[key] = value

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
    "b_size": 32,
    "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057])
}

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "az_bayes_reward_000"
config = getattr(config,config_name)

DEFAULT_CONFIG = {
    "rllib_config":{
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": 6,
        "sample_batch_size": 128,
        "train_batch_size": 128,
        "sgd_minibatch_size": 128,
        "lr": 1e-5,
        "num_sgd_iter": 3,
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 25,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.020,
            "dirichlet_noise": 0.003,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
            "policy_optimization": True,
        },
        "ranked_rewards": {
            "enable": True,
            "percentile": 75,
            "buffer_max_length": 1000,
            # add rewards obtained from random policy to
            # "warm start" the buffer
            "initialize_buffer": True,
            "num_init_rewards": 50,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
        },
        "evaluation_interval": 1,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 1,
        "num_cpus_per_worker": 1,
        "num_gpus": 1,
        "num_gpus_per_worker": 0.3,
        "callbacks": AZCallbacks # {"on_episode_end": LambdaZero.utils.dock_metrics},
    },
    "summaries_dir": summaries_dir,
    "memory": 60 * 10 ** 9,
    "trainer": AlphaZeroTrainer,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 2000000},
    "reward_learner_config": {
        "aq_size0": 3000,
        "data": dict(data_config, **{"dataset_creator":None}),
        "aq_size": 32,
        "kappa": 0.1,
        "sync_freq": 25,
        "epsilon": 0.0,
        "minimize_objective": False,
        "mol_dump_loc": "",
        "b_size": 32,
        'num_mol_retrain': 1000,
        "device": "cuda",
        "qed_cutoff": [0.2, 0.7],
        "synth_config": synth_config,
        'regressor_config': {
            "lambda": 6.16e-9,
            "data": dict(data_config, **{"dataset_creator":None}),
            "T": 20,
            "lengthscale": 1e-2,
            "uncertainty_eval_freq":15,
            "train_iterations": 64,
            "finetune_iterations": 16,
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
            "is_reward_model": True
        },
        "regressor": MCDrop,
    },
    "use_dock": False,
    "pretrained_model": None #  "/home/mjain/scratch/mcdrop_rl/model.pt"
}

config = merge_dicts(DEFAULT_CONFIG, config)

# convenience option to debug on maksym's personal laptop
machine = socket.gethostname()
if machine == "Ikarus":
    config["rllib_config"]["num_workers"] = 4
    config["rllib_config"]["memory"] = 25 * 10**9


if __name__ == "__main__":
    ray.init(memory=config["memory"])
    
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)
    
    mol_dump_loc = '/scratch/mjain/lambdabo_mol_dump/'
    curr_trial =  config_name + time.strftime("%Y-%m-%d_%H-%M-%S")
    
    if not osp.exists(osp.join(mol_dump_loc, curr_trial)):
        os.mkdir(osp.join(mol_dump_loc, curr_trial))
    config['reward_learner_config']['mol_dump_loc'] = osp.join(mol_dump_loc, curr_trial)
    
    reward_learner = BayesianRewardActor.remote(config['reward_learner_config'], 
                                                config["use_dock"], 
                                                config["rllib_config"]['env_config']['reward_config']['binding_model'],
                                                config["pretrained_model"])

    config['rllib_config']['env_config']['reward_config']['reward_learner'] = reward_learner
    config['rllib_config']['env_config']['reward_config']['regressor'] = config['reward_learner_config']['regressor']
    config['rllib_config']['env_config']['reward_config']['regressor_config'] = config['reward_learner_config']['regressor_config']
    config['rllib_config']['env_config']['reward_config']['kappa'] = config['reward_learner_config']['kappa']
    config['rllib_config']['env_config']['reward_config']['sync_freq'] = config['reward_learner_config']['sync_freq']
    #ModelCatalog.register_custom_model("MolActorCritic_tfv1", MolActorCritic_tfv1)
    # time.sleep(60)
    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])
