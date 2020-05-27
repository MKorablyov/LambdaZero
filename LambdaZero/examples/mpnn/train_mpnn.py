import socket, os, time
import numpy as np
import os.path as osp

from tqdm import tqdm

import torch as th
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import ray
from ray.rllib.models.catalog import ModelCatalog
from ray import tune

from LambdaZero.utils import get_external_dirs
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models



def train_epoch(loader, model, optimizer, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.train()
    sum_loss, sum_mse = 0,0
    for bidx,data in enumerate(loader):
        # compute y_hat and y
        data = data.to(device)
        optimizer.zero_grad()
        y_hat = model(data)
        y = getattr(data, target)

        loss = F.mse_loss(y_hat, (y - target_norm[0]) / target_norm[1])
        loss.backward()
        optimizer.step()
        mse = (((y_hat * target_norm[1]) + target_norm[0] - y)**2).mean()

        sum_loss += loss.item() * data.num_graphs
        sum_mse += mse.item() * data.num_graphs

    mean_loss, mean_mse = sum_loss / len(loader.dataset), sum_mse / len(loader.dataset)

    return {"mse" : mean_mse, "loss": mean_loss}


def eval_epoch(loader, model, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.eval()
    sum_loss, sum_mse = 0, 0
    for data in loader:
        # compute y_hat and y
        data = data.to(device)
        y_hat = model(data)
        y = getattr(data, target)

        loss = F.mse_loss(y_hat, (y - target_norm[0]) / target_norm[1])
        mse = (((y_hat * target_norm[1]) + target_norm[0] - y) ** 2).mean()

        sum_loss += loss.item() * data.num_graphs
        sum_mse += mse.item() * data.num_graphs

    mean_loss, mean_mse = sum_loss / len(loader.dataset), sum_mse / len(loader.dataset)
    return {"mse": mean_mse, "loss": mean_loss}


class BasicRegressor(tune.Trainable):
    def _setup(self, config):

        self.config = config
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # load dataset
        dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                               props=config["molprops"],
                                               transform=config["transform"],
                                               file_names=config["file_names"])

        # split dataset
        self.train_set = DataLoader(dataset[:8000], shuffle=True, batch_size=config["b_size"])
        self.val_set = DataLoader(dataset[8000:9000], batch_size=config["b_size"])
        self.test_set = DataLoader(dataset[9000:], batch_size=config["b_size"])

        # make model
        self.model = LambdaZero.models.MPNNet()
        self.model.to(self.device)
        self.optim = th.optim.Adam(self.model.parameters(), lr=config["lr"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.train_set, self.model,  self.device, self.config)

        return eval_scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        th.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(th.load(checkpoint_path))


transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()



DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "dataset_root": os.path.join(datasets_dir, "brutal_dock/d4"),
        "targets": ["gridscore"],
        "target_norms": [[-26.3, 12.3]],
        "file_names": ["dock_blocks105_walk40_clust"],
        "transform": transform,

        "lr": 0.001,
        "b_size": 128,
        "dim": 64,
        "num_epochs": 120,

        #"model": "some_model", todo

        "molprops": ["gridscore", "klabel"],
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        # todo: test epoch
        },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 250000000,
    "stop": {"training_iteration": 2},
}

config = DEFAULT_CONFIG


if __name__ == "__main__":
    ray.init()

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop={"training_iteration":100}, #EarlyStop(),
                        resources_per_trial={
                           "cpu": 4, # fixme cpu allocation could block inputs ....
                           "gpu": 1.0
                        },
                        num_samples=1,
                        checkpoint_at_end=False,
                        local_dir=summaries_dir,
                        checkpoint_freq=100000)