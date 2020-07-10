import socket, os,sys, time
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

import ray
from ray import tune
from ray.rllib.utils import merge_dicts

from LambdaZero.utils import get_external_dirs, BasicRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.mpnn import config

transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "mpnn000"
config = getattr(config, config_name)

def train_epoch(loader, model, optimizer, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss":0, "mse": 0, "mae":0}
    for bidx,data in enumerate(loader):
        data = data.to(device)
        target = getattr(data, config["target"])

        optimizer.zero_grad()
        logit = model(data, do_dropout=True)
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

    # todo: take a bunch of samples here
    # todo: concatenate a lot of stuff
    # todo: compute mean/variance per sample
    # todo: compute error per sample


    # todo: estimate correlation of stochastic variable


    for bidx, data in enumerate(loader):
        data = data.to(device)
        target = getattr(data, config["target"])

        logit = model(data,do_dropout=True)
        loss = F.mse_loss(logit, normalizer.normalize(target))

        metrics["loss"] += loss.item() * data.num_graphs
        pred = normalizer.unnormalize(logit)
        metrics["mse"] += ((target - pred) ** 2).sum().item()
        metrics["mae"] += ((target - pred).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)



    return metrics


DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "target": "gridscore",
        "target_norm": [-26.3, 12.3],
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/d4/raw/randsplit_dock_blocks105_walk40_clust.npy"),
        "b_size": 64,

        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/d4"),
            "props": ["gridscore", "klabel"],
            "transform": transform,
            "file_names": ["dock_blocks105_walk40_clust"],
        },

        "model": LambdaZero.models.MPNNetDrop,
        "model_config": {},

        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },

    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,

    "stop": {"training_iteration": 3},
    "resources_per_trial": {
        "cpu": 4,  # fixme requesting all CPUs blocks additional call to ray from LambdaZero.input
        "gpu": 1.0
    },
    "num_samples":1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 250000000,
}


config = merge_dicts(DEFAULT_CONFIG, config)


if __name__ == "__main__":
    ray.init(memory=config["memory"])

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"])