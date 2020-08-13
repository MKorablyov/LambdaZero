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

from math import sqrt, pi as PI
from dimenet import DimeNet

from LambdaZero.utils import get_external_dirs, BasicRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.mpnn import config

transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if len(sys.argv) >= 2:
    config_name = sys.argv[1]
    if len(sys.argv) > 2:
        if len(sys.argv) > 3:
            split_type = model_type = sys.argv[3]
        else:
            split_type = "rand"
        model_type = sys.argv[2]
    else:
        model_type = "mpnn"
        split_type = "rand"
else:
    config_name = "mpnn000"
    model_type = "mpnn"
    split_type = "rand"

config = getattr(config, config_name)

def train_epoch(loader, model, optimizer, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss":0,"mae":0,"mse":0,"top15_regret":0,"top50_regret":0}
    epoch_targets = []
    epoch_preds = []
    
    i = 0
    total_iters = int(len(loader.dataset)/config["b_size"])

    for bidx,data in enumerate(loader):

        i += 1
        print(str(i) + "/" + str(total_iters))

        targets = getattr(data, config["target"]).to(device)

        optimizer.zero_grad()

        if config["model_type"] == "dime":
            z = data.z.to(device)
            pos = data.pos.to(device)
            batch = data.batch.to(device)

            logits = model(z=z,pos=pos,batch=batch).squeeze(1)
        else:
            data = data.to(device)
            logits = model(data)

        if config["loss"] == "L2":
            loss = F.mse_loss(logits, normalizer.normalize(targets))
        else:
            loss = F.l1_loss(logits, normalizer.normalize(targets))

        loss.backward()
        optimizer.step()

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.unnormalize(logits).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets,0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])
    return metrics


def eval_epoch(loader, model, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.eval()
    metrics = {"loss":0,"mae":0,"mse":0,"top15_regret":0,"top50_regret":0}
    epoch_targets = []
    epoch_preds = []

    for bidx,data in enumerate(loader):
        targets = getattr(data, config["target"]).to(device)
        if config["model_type"] == "dime":
            z = data.z.to(device)
            pos = data.pos.to(device)
            batch = data.batch.to(device)

            logits = model(z=data.z,pos=data.pos,batch=data.batch).squeeze(1)
        else:
            data = data.to(device)

            logits = model(data)

        loss = F.mse_loss(logits, normalizer.normalize(targets))

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.unnormalize(logits).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets,0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])

    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])
    return metrics

loaded_model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3) if model_type == "dime" else LambdaZero.models.MPNNet()

split_file = "ksplit_Zinc15_260k.npy" if split_type == "KNN" else "randsplit_Zinc15_260k.npy"

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "target": "gridscore",
        "target_norm": [-43.042, 7.057],
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/" + split_file),
        "b_size": 128,

        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore"],
            "transform": transform,
            "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]
        },
        "model": loaded_model,
        "model_config": {},

        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "model_type": model_type
    },

    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,

    "stop": {"training_iteration": 100},
    "resources_per_trial": {
        "cpu": 12,  # fixme - calling ray.remote would request resources outside of tune allocation
        "gpu": 1.0
    },
    "keep_checkpoint_num": 1,
    "checkpoint_score_attr":"train_loss",
    "num_samples":1,
    "checkpoint_at_end": True,
    "checkpoint_freq": 100000,
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
                        local_dir=summaries_dir+"/dime/",
                        name=config_name,
                        checkpoint_freq=config["checkpoint_freq"])
