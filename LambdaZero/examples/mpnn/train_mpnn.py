import socket, os, time
import numpy as np
import os.path as osp
import torch as th
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
import ray
from ray import tune

from LambdaZero.utils import get_external_dirs
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models


def train_epoch(loader, model, optimizer, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss":0, "mse": 0, "mae":0}
    for bidx,data in enumerate(loader):
        data = data.to(device)
        target = getattr(data, config["target"])

        optimizer.zero_grad()
        logit = model(data)
        loss = F.mse_loss(logit, normalizer.normalize(target))
        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item() * data.num_graphs
        pred = normalizer.unnormalize(logit)
        metrics["mse"] += ((target - pred) ** 2).sum().item()
        metrics["mae"] += ((target - pred).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    return metrics


def eval_epoch(loader, model, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for bidx, data in enumerate(loader):
        data = data.to(device)
        target = getattr(data, config["target"])

        logit = model(data)
        loss = F.mse_loss(logit, normalizer.normalize(target))

        metrics["loss"] += loss.item() * data.num_graphs
        pred = normalizer.unnormalize(logit)
        metrics["mse"] += ((target - pred) ** 2).sum().item()
        metrics["mae"] += ((target - pred).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    return metrics



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
        split_path = osp.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
        train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)

        self.train_set = DataLoader(Subset(dataset, train_idxs.tolist()), shuffle=True, batch_size=config["b_size"])
        self.val_set = DataLoader(Subset(dataset, val_idxs.tolist()), batch_size=config["b_size"])
        self.test_set = DataLoader(Subset(dataset, test_idxs.tolist()), batch_size=config["b_size"])

        # make model
        self.model = LambdaZero.models.MPNNet()
        self.model.to(self.device)
        self.optim = th.optim.Adam(self.model.parameters(), lr=config["lr"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.val_set, self.model,  self.device, self.config)
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
        "target": "gridscore",
        "target_norm": [-26.3, 12.3],
        "file_names": ["dock_blocks105_walk40_clust"],
        "transform": transform,
        "split_name": "randsplit_dock_blocks105_walk40_clust",
        "lr": 0.001,
        "b_size": 64,
        "dim": 64,
        "num_epochs": 120,

        #"model": "some_model", todo
        # optimizer:
        # lr = :

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