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

@ray.remote
class Member:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LambdaZero.models.MPNNetDrop(**self.config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](self.model.parameters(), **config["optimizer_config"])

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def predict(self, loader):
        preds = []
        for batch in loader:
            preds.extend(self.model(batch.to(self.device), do_dropout=False).cpu().detach().numpy().tolist())
        return np.array(preds)

    def train(self, train_loader, val_loader, from_scratch=False, validate=False):
        # import pdb; pdb.set_trace()
        if from_scratch:
            self.model = LambdaZero.models.MPNNetDrop(**self.config["model_config"])
            self.model.to(self.device)
            self.optim = self.config["optimizer"](self.model.parameters(), **self.config["optimizer_config"])
        all_scores = []
        for i in range(self.config["train_iterations"]):
            scores = self.config["train"](train_loader, val_loader, self.model, self.device, self.config, self.optim, i)
            all_scores.append(scores)
        
        if validate:
            val_score = self.config["eval_epoch"](val_loader, self.model, self.device, self.config, "val")
            return all_scores[-1], {**val_score, **val_ll}
        else:
            return all_scores[-1], {}

class DeepEnsemble(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.val_loader = None
        if config["data"]["dataset_creator"] is not None:
            self.train_loader, self.val_loader = config["data"]["dataset_creator"](config["data"])

        # make model
        # self.model = LambdaZero.models.MPNNetDrop(**self.config["model_config"])
        # self.model.to(self.device)
        # import pdb; pdb.set_trace()
        # self.optim = config["optimizer"](self.model.parameters(), **config["optimizer_config"])
        self.models = [Member.remote(self.config) for _ in range(config["num_members"])]
        # import pdb; pdb.set_trace()

    def get_weights(self):
        return [ray.get(model.get_weights.remote()) for model in self.models]
        # return self.model.state_dict()

    def set_weights(self, weights):
        [self.models[i].set_weights.remote(weights[i]) for i in range(len(weights))]
        # self.model.load_state_dict(weights)

    def fit(self, train_loader, val_loader, validate=False):
        # scores = {"model_" + str(i): ray.get(self.models[i].train.remote(train_loader, val_loader, True, validate)) for i in range(len(self.models))}
        # return scores, {}
        runs = [model.train.remote(train_loader, val_loader, True, validate) for model in self.models]
        while True:
            done, wait = ray.wait(runs)
            print(done)
            if len(wait) == 0:
                break
        scores = {}
        # scores["mod_1"] = ray.get(self.models[0].train.remote(train_loader, val_loader, True, validate))
        # scores["mod_2"] = ray.get(self.models[1].train.remote(train_loader, val_loader, True, validate))
        # if from_scratch:
        # self.model = LambdaZero.models.MPNNetDrop(**self.config["model_config"])
        # self.model.to(self.device)
        # self.optim = self.config["optimizer"](self.model.parameters(), **self.config["optimizer_config"])
        # all_scores = []
        # for i in range(self.config["train_iterations"]):
        #     scores = self.config["train"](train_loader, val_loader, self.model, self.device, self.config, self.optim, i)
        #     all_scores.append(scores)
        
        # if validate:
        #     val_score = self.config["eval_epoch"](val_loader, self.model, self.device, self.config, "val")
        #     return all_scores[-1], {**val_score, **val_ll}
        # else:
        #     return all_scores[-1], {}
    
    def save(self, path):
        for i in range(len(self.models)):
            torch.save(ray.get(self.models[i].get_weights.remote()), os.path.join(path, f'model-{i}.pt'))
        # torch.save(ray.get(self.get_weights(), os.path.join(path, f'model-{i}.pt')))

    def get_mean_variance(self, loader, train_len=None):
        preds = np.vstack([ray.get(model.predict.remote(loader)) for model in self.models]).T
        # preds = []
        # for batch in loader:
        #     preds.extend(self.model(batch).cpu().detach().numpy().tolist())
        # preds = np.array(preds)
        mean = np.mean(preds, axis=-1)
        var = np.var(preds, axis=-1)
        # import pdb;pdb.set_trace()
        return mean, var


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
        "run_or_experiment": DeepEnsemble,
        "config": {
            "data":data_config,
            "lambda": 1e-11,
            "num_members": 3,
            "lengthscale": 1e-2,
            "uncertainty_eval_freq":15,
            "train_iterations":150,
            "model": LambdaZero.models.MPNNetDrop,
            "model_config": {"drop_data":False, "drop_weights":False, "drop_last":True, "drop_prob":0.1},
            # "model_config": {},

            "optimizer": torch.optim.Adam,
            "optimizer_config": {
                "lr": 0.001
            },

            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 61},
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

    ray.init(memory=config["memory"])
    tune.run(**config["regressor_config"])
