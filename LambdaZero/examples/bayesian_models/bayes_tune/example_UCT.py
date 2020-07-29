import time
import os.path as osp
import numpy as np
import torch

from ray import tune
from torch_geometric.data import DataLoader
from torch.utils.data import Subset, ConcatDataset

from example_mcdrop import DEFAULT_CONFIG as config




class UCT(tune.Trainable):
    def _setup(self, config):

        # acquired_dataset =
        # query_dataset =

        self.config = config
        # load dataset
        print(config)
        time.sleep(100)

        dataset = config["dataset"](**config["dataset_config"])

        train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)
        self.train_set = Subset(dataset, train_idxs.tolist())
        self.val_set = Subset(dataset, val_idxs.tolist())

        print(self.train_set)


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







if __name__ == "__main__":
    uct = UCT(config["regressor_config"])

