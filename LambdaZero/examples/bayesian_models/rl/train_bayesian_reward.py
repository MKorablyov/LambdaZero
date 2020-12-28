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

# from LambdaZero.environments import block_mol_v3
from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop, MCDropGenAcqf
from LambdaZero.examples.bayesian_models.rl import config
from LambdaZero.environments.reward import BayesianRewardActor

# from LambdaZero.examples.bayesian_models.bayes_tune import config
from LambdaZero.examples.bayesian_models.bayes_tune.functions import get_tau, train_epoch_with_targets, eval_epoch, \
    train_mcdrop_rl, mcdrop_mean_variance
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config

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
else: config_name = "ppo_bayes_reward_general_acqf_UCB"
config = getattr(config,config_name)

DEFAULT_CONFIG = {
    "rllib_config":{
        "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
        "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
        "num_workers": 8,
        "num_gpus_per_worker": 0.25,
        "num_gpus": 1,
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
        },
        "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics}, # fixme (report all)
        "framework": "torch",
        "lr": 5e-5,
    },
    "summaries_dir": summaries_dir,
    "memory": 60 * 10 ** 9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 2000000},
    "reward_learner_config": {
        "aq_size0": 3000,
        "data": dict(data_config, **{"dataset_creator":None}),
        "aq_size": 32,
        "mol_dump_loc": "",
        "kappa": 0.1,
        "sync_freq": 50,
        "epsilon": 0.0,
        "minimize_objective": False,
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
            "train_iterations": 72,
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
        "regressor": MCDropGenAcqf,
    },
    "use_dock": False,
    "pretrained_model": None #  "/home/mjain/scratch/mcdrop_rl/model.pt"
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
    mol_dump_loc = '/home/nekoeiha/scratch/Summaries/lambdabo_mol_dump/'
    curr_trial =  config_name + time.strftime("%Y-%m-%d_%H-%M-%S")
    
    if not osp.exists(osp.join(mol_dump_loc, curr_trial)):
        os.makedirs(osp.join(mol_dump_loc, curr_trial))
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

    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])
