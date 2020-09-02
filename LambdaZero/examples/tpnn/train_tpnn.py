import os
import sys
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

import ray
from ray import tune
from ray.rllib.utils import merge_dicts

from LambdaZero.utils import get_external_dirs, TPNNRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.tpnn import config

proc_func = LambdaZero.inputs.tpnn_proc
transform = LambdaZero.inputs.tpnn_transform
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

config_name = sys.argv[1] if len(sys.argv) >= 2 else "tpnn_default"
config = getattr(config, config_name)


def train_epoch(loader, model, optimizer, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss": 0}
    epoch_targets = []
    epoch_preds = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        logits = model(data).view(-1)
        loss = F.mse_loss(logits, normalizer.tfm(targets))
        loss.backward()
        optimizer.step()

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.itfm(logits).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets, 0)
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
    metrics = {"loss": 0}
    epoch_targets = []
    epoch_preds = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        logits = model(data).view(-1)
        loss = F.mse_loss(logits, normalizer.tfm(targets))

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.itfm(logits).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets, 0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])
    return metrics


TPNN_CONFIG = {
    "trainer": TPNNRegressor,
    "trainer_config": {
        "target": "y",  # gridscore
        "target_norm": [-49.3, 26.1],  # [-43.042, 7.057],
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),  # osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
        "b_size": 64,

        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore", "coord"],
            "proc_func": proc_func,
            "transform": transform,
            "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],  # ["Zinc15_2k"]
        },

        "model": LambdaZero.models.TPNN_v2,
        "model_config": {},
        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },

    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,

    "stop": {"training_iteration": 120},
    "resources_per_trial": {
        "cpu": 4,  # fixme - calling ray.remote would request resources outside of tune allocation
        "gpu": 1.0
    },
    "keep_checkpoint_num": 2,
    "checkpoint_score_attr": "train_loss",
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 100000,  # 1;  0 is a default in tune.run
}


config = merge_dicts(TPNN_CONFIG, config)


if __name__ == "__main__":
    ray.init(memory=config["memory"])

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"],
                        name=config_name)
