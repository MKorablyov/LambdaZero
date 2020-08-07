import time, random
import os.path as osp
import numpy as np
import torch
import ray
from ray import tune
from torch_geometric.data import DataLoader
from torch_geometric import transforms as T
from torch.utils.data import Subset, ConcatDataset
import LambdaZero.utils
import LambdaZero.models
import LambdaZero.inputs
from LambdaZero.examples.bayesian_models.bayes_tune.example_mcdrop import MCDrop
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_epoch,eval_epoch, train_mcdrop

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


class UCT(tune.Trainable):
    def _setup(self, config):
        self.config = config
        # load dataset
        self.dataset = config["dataset"](**config["dataset_config"])
        ul_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)

        np.random.shuffle(ul_idxs) # randomly acquire batch zero
        train_idxs = ul_idxs[:self.config["b_size0"]]
        ul_idxs = ul_idxs[self.config["b_size0"]:]

        train_set = Subset(self.dataset, train_idxs.tolist())
        ul_set = Subset(self.dataset, ul_idxs.tolist())
        val_set = Subset(self.dataset, val_idxs.tolist())
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=config["b_size"])
        self.ul_loader = DataLoader(ul_set, batch_size=config["b_size"])
        self.val_loader = DataLoader(val_set, batch_size=config["b_size"])

        # make model with uncertainty
        self.regressor = self.config["regressor"](**config["regressor_config"])
        self.regressor.fit(self.train_loader,self.val_loader)

    def _train(self):
        idxs = self.acquire_batch()
        scores = self.update_with_seen(idxs)
        return scores

    def update_with_seen(self, idxs):
        print(len(self.train_loader.dataset), len(self.ul_loader.dataset), len(self.val_loader.dataset))

        # update train/unlabeled datasets
        aq_mask = np.zeros(len(self.ul_loader.dataset.indices),dtype=np.bool)
        aq_mask[idxs] = True
        aq_idxs = np.asarray(self.ul_loader.dataset.indices)[aq_mask].tolist()
        ul_idxs = np.asarray(self.ul_loader.dataset.indices)[~aq_mask].tolist()
        train_idxs = self.train_loader.dataset.indices + aq_idxs
        train_set = Subset(self.dataset, train_idxs)
        ul_set = Subset(self.dataset, ul_idxs)
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config["b_size"])
        self.ul_loader = DataLoader(ul_set, batch_size=self.config["b_size"])
        # fit model to the data
        scores = self.regressor.fit(self.train_loader, self.val_loader)
        return scores[-1]

    def acquire_batch(self):
        mean, var = self.regressor.get_mean_variance(self.ul_loader)
        scores = mean + (self.config["kappa"] * var)
        if self.config["invert_objective"]: scores = -scores
        idxs = np.argsort(-scores)[:self.config["b_size"]]
        return idxs


transform = T.Compose([LambdaZero.utils.Complete()])
regressor_config = {
    "config":{
        "target": "gridscore",
        "dataset_creator": None, #LambdaZero.utils.dataset_creator_v1,
        "b_size": 40,
        "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057]),
        "lambda": 1e-8,
        "T": 20,
        "drop_p": 0.1,
        "lengthscale": 1e-2,
        "uncertainty_eval_freq": 15,
        "model": LambdaZero.models.MPNNetDrop,
        "model_config": {},
        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "train": train_mcdrop
    }
}
DEFAULT_CONFIG = {
    "acquirer_config": {
        "run_or_experiment": UCT,
        "config":{
            "dataset_creator": LambdaZero.utils.dataset_creator_v1,
            "dataset_split_path": osp.join(datasets_dir,
                                           "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
            # "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
            "dataset": LambdaZero.inputs.BrutalDock,
            "dataset_config": {
                "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
                "props": ["gridscore", "smi"],
                "transform": transform,
                "file_names":
                    ["Zinc15_2k"],
                # ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
            },
            "regressor": MCDrop,
            "regressor_config": regressor_config,
            "b_size0": 200,
            "b_size": 20,
            "kappa": 0.2,
            "invert_objective":True,
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 200},
        "resources_per_trial": {"cpu": 4, "gpu": 1.0}
    },
    "memory": 10 * 10 ** 9
}


config = DEFAULT_CONFIG
if __name__ == "__main__":
    ray.init(memory=config["memory"])
    tune.run(**config["acquirer_config"])