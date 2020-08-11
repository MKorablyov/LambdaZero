import time
import os.path as osp
import numpy as np
import torch


import ray
from ray import tune
from torch_geometric.data import DataLoader
from torch.utils.data import Subset, ConcatDataset
from LambdaZero.utils import get_external_dirs

from LambdaZero.examples.bayesian_models.bayes_tune.example_mcdrop import DEFAULT_CONFIG as mcdrop_config

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class UCT(tune.Trainable):

    def _setup(self, config):

        # acquired_dataset =
        # query_dataset =

        self.config = config
        print(config.keys())

        #print(self.train_set)
        time.sleep(100)

        # load dataset
        dataset = config["dataset"](**config["dataset_config"])
        train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)
        self.train_set = Subset(dataset, train_idxs.tolist())
        self.val_set = Subset(dataset, val_idxs.tolist())
        #self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=config["b_size"])
        #self.val_loader = DataLoader(self.val_set, batch_size=config["b_size"])
        #self.regressor = config["regressor"](**config["regressor_config"])



    def _train(self):
        # idxs = self.acquire_batch()
        # self.update_with_seen(idxs)
        pass

    def update_with_seen(self, dataset):
        # if_acquired[idxs] = True
        # model.fit(data(Subset([if_acquired])))
        pass

    def acquire_batch(self):
        # model.predict_mean_variance(self.query_dataset)
        # scores = mean + (kappa * variance)
        pass




#mcdrop_config = mcdrop_config["regressor_config"]
#dataset = mcdrop_config["config"]["dataset"]


DEFAULT_CONFIG = {
    "acquirer_config": {
        "run_or_experiment": UCT,
        "config":{
            "regressor_config": mcdrop_config, # todo: do not load the dataset
            "b_size": 10,
            "kappa": 1.0,
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
