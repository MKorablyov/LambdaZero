import socket, os, time
import numpy as np
import os.path as osp
import math
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
#from torch_geometric.data import DataLoader
from torch.utils.data import Sampler
import time

from torch_geometric.data import Data, Batch
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T
import ray
from ray import tune

from LambdaZero.utils import get_external_dirs
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models

from custom_dataloader import DL

np.set_printoptions(threshold=10)

def train_epoch(loader, model, optimizer, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.train()

    metrics = {"loss":0, "mse": 0, "mae":0}

    i = 0
    for bidx,data in enumerate(loader):
        print(data)
        print(bidx, i)
        i += 1
        # compute y_hat and y
        data = data.to(device)

        optimizer.zero_grad()
        logit = model(data)
        pred = (logit * target_norm[1]) + target_norm[0]
        y = getattr(data, target)

        loss = F.mse_loss(logit, (y - target_norm[0]) / target_norm[1])
        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item() * data.num_graphs
        metrics["mse"] += ((y - pred) ** 2).sum().item()
        metrics["mae"] += ((y - pred).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    return metrics


def eval_epoch(loader, model, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for bidx, data in enumerate(loader):
        # compute y_hat and y
        data = data.to(device)
        logit = model(data)
        pred = (logit * target_norm[1]) + target_norm[0]
        y = getattr(data, target)

        loss = F.mse_loss(logit, (y - target_norm[0]) / target_norm[1])
        metrics["loss"] += loss.item() * data.num_graphs
        metrics["mse"] += ((y - pred) ** 2).sum().item()
        metrics["mae"] += ((y - pred).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    return metrics


class BasicRegressor(tune.Trainable):
    def _setup(self, config):

        self.config = config
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # make model
        self.model = LambdaZero.models.MPNNet()
        self.model.to(self.device)
        self.optim = th.optim.Adam(self.model.parameters(), lr=config["lr"])

        # load dataset
        dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                               props=config["molprops"],
                                               transform=config["transform"],
                                               file_names=config["file_names"])

        # split dataset
        split_path = osp.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
        train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)

        bsize = config["b_size"]
        train_dataset = dataset[th.tensor(train_idxs)]

        energies = []  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
        for i in range(len(train_idxs)):
            energies.append(abs(train_dataset[i].gridscore))
        energies = th.tensor(energies).double()
        sampler = th.utils.data.sampler.WeightedRandomSampler(energies, len(energies), replacement=True)

        train_subset = Subset(dataset, train_idxs.tolist())
        val_subset = Subset(dataset, val_idxs.tolist())
        test_subset = Subset(dataset, test_idxs.tolist())

        self.train_set = DL(train_subset, batch_size=bsize, samp=sampler)
        self.val_set = DL(val_subset, batch_size=bsize)
        self.test_set = DL(test_subset, batch_size=bsize)

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.train_set, self.model,  self.device, self.config)
        # rename to make scope
        train_scores = [("train_" + k, v) for k,v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

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
        "split_name": "randsplit_dock_blocks105_walk40_clust",
        "lr": 0.001,
        "b_size": 64,
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
                           "cpu": 4, # fixme requesting all CPUs blocks additional call to ray from LambdaZero.input
                           "gpu": 1.0
                        },
                        num_samples=1,
                        checkpoint_at_end=False,
                        local_dir=summaries_dir,
                        checkpoint_freq=100000)