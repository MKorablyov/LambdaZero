import socket, os,sys, time, random
import numpy as np
import os.path as osp
import torch
from torch_geometric import transforms as T

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.utils import merge_dicts
from ray.rllib.models.catalog import ModelCatalog
from sklearn import linear_model
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.bayesian_models.bayes_tune import config
from LambdaZero.examples.bayesian_models.bayes_tune.functions import get_tau, train_epoch, eval_epoch, \
    train_mcdrop, mcdrop_mean_variance, eval_mcdrop, mpnn_brr_mean_variance


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
transform = T.Compose([LambdaZero.utils.Complete()])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MCDrop(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.val_loader = None
        if config["data"]["dataset_creator"] is not None:
            self.train_loader, self.val_loader = config["data"]["dataset_creator"](config["data"])

        # self.feature_dim = 64
        # make model
        self.model = LambdaZero.models.MPNNetDrop(**self.config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](self.model.parameters(), **config["optimizer_config"])
        self.train_len = 0

    def _train(self): # this is train iteration actually
        stats = self.config["train"](
            self.train_loader, self.val_loader, self.model,self.device, self.config, self.optim, self._iteration)
        return stats

    # def fine_tune(self, new_samples_loader, val_loader, validate):
    #     self.train_loader, self.val_loader = new_samples_loader, val_loader
    #     self.train_len += len(self.train_loader.dataset)
    #     all_scores = []
    #     for i in range(self.config["finetune_iterations"]):
    #         self._iteration = i
    #         scores = self._train()
    #         all_scores.append(scores)
    #     # self.train_loader = train_loader
    #     if validate:
    #         val_score = self.config["eval_epoch"](self.val_loader, self.model, self.device, self.config, "val")
    #         val_ll = eval_mcdrop(self.val_loader, self.model, self.device, self.config, self.train_len, "val")
    #         return all_scores[-1], {**val_score, **val_ll}
    #     else:
    #         return all_scores[-1], {}

    def fit(self, train_loader, val_loader, validate=False):
        # update internal dataset
        self.train_loader, self.val_loader = train_loader, val_loader
        self.train_len = len(train_loader.dataset)
        # make a new model
        self.model = LambdaZero.models.MPNNetDrop(**self.config["model_config"]) # todo: reset?
        self.optim = self.config["optimizer"](self.model.parameters(), **self.config["optimizer_config"])
        self.model.to(self.device)

        # todo allow ray native stopping
        all_scores = []
        for i in range(self.config["train_iterations"]):
            self._iteration = i
            scores = self._train()
            all_scores.append(scores)
        return all_scores[-1], {}
        # if validate:
        #     val_score = self.config["eval_epoch"](self.val_loader, self.model, self.device, self.config, "val")
        #     val_ll = eval_mcdrop(self.val_loader, self.model, self.device, self.config, self.train_len, "val")
        #     return all_scores[-1], {**val_score, **val_ll}
        # else:
        #     return all_scores[-1], {}

    def get_mean_variance(self, loader, train_loader):
        mean,var = self.config["get_mean_variance"](train_loader, loader, self.model, self.device, self.config)
        return mean, var

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, f'model.pt'))

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)


data_config = {
    "target": "gridscore",
    "dataset_creator": LambdaZero.inputs.dataset_creator_v1,
    "dataset_split_path": osp.join(datasets_dir,
                                   "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
                                   #"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
    "dataset": LambdaZero.inputs.BrutalDock,
    "dataset_config": {
        "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
        "props": ["gridscore", "smi"],
        "transform": transform,
        "file_names": ["Zinc15_2k"],
        #["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
    },
    "b_size": 40,
    "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057])
}


DEFAULT_CONFIG = {
    "regressor_config":{
        "run_or_experiment": MCDrop,
        "config": {
            "data":data_config,
            "lambda": 1e-11,
            "T": 20,
            "lengthscale": 1e-2,
            "uncertainty_eval_freq":1500,
            "train_iterations": 151,
            "model": LambdaZero.models.MPNNetDrop,
            "model_config": {"drop_data":False, "drop_weights":False, "drop_last":True, "drop_prob":0.1,},
            # "model_config": {},

            "optimizer": torch.optim.Adam,
            "optimizer_config": {
                "lr": 0.001
            },

            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 151},
        "resources_per_trial": {
            "cpu": 4,
            "gpu": 1.0
        },
        # "keep_checkpoint_num": 2,
        # "checkpoint_score_attr":"train_loss",
        # "num_samples":1,
        # "checkpoint_at_end": False,
    },
    "memory": 10 * 10 ** 9
}



if __name__ == "__main__":
    if len(sys.argv) >= 2: config_name = sys.argv[1]
    else: config_name = "mcdrop001"
    config = getattr(config, config_name)
    config = merge_dicts(DEFAULT_CONFIG, config)
    config["regressor_config"]["name"] = config_name

    ray.init(_memory=config["memory"])
    # this will run train the model with tune scheduler
    #tune.run(**config["regressor_config"])
    tune.run(**config["regressor_config"])
#     # this will fit the model outside of tune
    # mcdrop = MCDrop(config["regressor_config"]["config"])
    # print(mcdrop.fit(mcdrop.train_loader, mcdrop.val_loader))
    # print(mcdrop.get_mean_variance(mcdrop.train_loader))