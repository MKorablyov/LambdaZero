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
from LambdaZero.environments import PredDockBayesianReward_v1


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "ppo_bayes_reward_008"
config = getattr(config,config_name)
curr_trial = config_name + time.strftime("%Y-%m-%d_%H-%M-%S")


data_config = {
    "target": "dockscore",
    #"dataset_creator": LambdaZero.inputs.dataset_creator_v1,
    "dataset_split_path": osp.join(datasets_dir,
                                #    "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
                                #"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
                                "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
    "dataset": LambdaZero.inputs.BrutalDock,
    "dataset_config": {
        "root": osp.join(datasets_dir, "brutal_dock/seh/raw"),
        "props": ["dockscore", "smiles"],
        "transform": T.Compose([LambdaZero.utils.Complete()]),
        "file_names": "Zinc20_docked_neg_randperm_3k.feather",
    },


    "b_size": 40,
    "target_norm": [-8.6, 1.10], # fixme use normalizer everywhere
    "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-8.6, 1.10])
}

regressor_config = {
    "lambda": 6.16e-9,
    "data": dict(data_config, **{"dataset_creator": None}),
    "T": 20,
    "lengthscale": 1e-2,
    "uncertainty_eval_freq": 15,
    "train_iterations": 72,
    "finetune_iterations": 16,
    "model": LambdaZero.models.MPNNetDrop,
    # fixme !!!!!!!! this default model only does drop in the last layer
    "model_config": {"drop_data": False, "drop_weights": False, "drop_last": True, "drop_prob": 0.1},
    "optimizer": torch.optim.Adam,
    "optimizer_config": {
        "lr": 0.001
    },
    "train_epoch": train_epoch_with_targets,
    "eval_epoch": eval_epoch,
    "train": train_mcdrop_rl,
    "get_mean_variance": mcdrop_mean_variance,
    "is_reward_model": True
}

reward_learner_config = {
    "reward_actor_cpus": 4,
    "reward_actor_gpus": 1.0,
    "aq_size0": 3000,
    "data": dict(data_config, **{"dataset_creator": None}),
    "aq_size": 32,
    "mol_dump_loc": osp.join(summaries_dir, curr_trial),
    "docking_loc": osp.join(summaries_dir, curr_trial, "docking"),
    "kappa": 0.1,
    "sync_freq": 50,
    "epsilon": 0.0,
    "minimize_objective": False,
    "b_size": 32,
    'num_mol_retrain': 1000,
    "device": "cuda",
    "qed_cutoff": [0.2, 0.7],
    "synth_config": synth_config,
    "regressor": MCDrop,
    "regressor_config": regressor_config
}

rllib_config = {
    #"tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
    #"local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
    "env": #BlockMolEnvGraph_v1,
          BlockMolGraphEnv_PersistentBuffer, # fixme maybe ray.remote the buffer as well
    "env_config": {
        "random_steps": 4,
        "allow_removal": True,
        "reward": PredDockBayesianReward_v1,
        "reward_config": {
            "synth_config": synth_config,
            "binding_model": osp.join(datasets_dir, "brutal_dock/seh/trained_weights/vanilla_mpnn/model.pth"),
            "dense_rewards":False,
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
    "reward_learner_config":reward_learner_config,
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

    config['reward_learner_config']['reward_actor_cpus'] = 1
    config['reward_learner_config']['reward_actor_gpus'] = 0.25
    config["reward_learner_config"]["num_mol_retrain"] = 25
    config["reward_learner_config"]["aq_size0"] = 14
    config["reward_learner_config"]["aq_size"] = 2
    config["reward_learner_config"]["b_size"] = 2
    config["reward_learner_config"]["data"]["b_size"] = 2

    config["reward_learner_config"]["regressor_config"]["train_iterations"] = 2
    config["reward_learner_config"]["regressor_config"]["finetune_iterations"] = 2
    config["reward_learner_config"]["regressor_config"]["T"] = 2


if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)
    #
    if not osp.exists(config["reward_learner_config"]['mol_dump_loc']):
        os.mkdir(config["reward_learner_config"]['mol_dump_loc'])
    if not osp.exists(config["reward_learner_config"]['docking_loc']):
        os.mkdir(config["reward_learner_config"]['docking_loc'])

    reward_learner = BayesianRewardActor.options(
        num_cpus=config["reward_learner_config"]['reward_actor_cpus'],
        num_gpus=config["reward_learner_config"]['reward_actor_gpus']).\
        remote(config["reward_learner_config"], config["use_dock"],
               config["rllib_config"]['env_config']['reward_config']['binding_model'], config["pretrained_model"])


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