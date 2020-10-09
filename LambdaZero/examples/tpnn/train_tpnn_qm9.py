import os
import sys
import numpy as np
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

import torch_geometric
from LambdaZero.inputs.inputs_op import atomic_num_to_group, atomic_num_to_period, N_GROUPS, N_PERIODS

datasets_dir, _, summaries_dir = get_external_dirs()

config_name = sys.argv[1] if len(sys.argv) >= 2 else "tpnn_default"
config = getattr(config, config_name)


def tpnn_transform_qm9(data):
    def _group_period(atomic_numbers):
        groups = torch.tensor([atomic_num_to_group[atomic_number] for atomic_number in atomic_numbers.tolist()])
        periods = torch.tensor([atomic_num_to_period[atomic_number] for atomic_number in atomic_numbers.tolist()])
        return groups, periods

    def _one_hot_group_period(groups, periods):
        groups_one_hot = torch.nn.functional.one_hot(groups - 1, N_GROUPS)
        periods_one_hot = torch.nn.functional.one_hot(periods - 1, N_PERIODS)
        return torch.cat((groups_one_hot, periods_one_hot), dim=1).type(torch.float64)

    def _rel_vectors(coordinates, bonds):
        origin_pos = coordinates[bonds[0]]
        neighbor_pos = coordinates[bonds[1]]
        return neighbor_pos - origin_pos

    data = data.clone()
    data.x = _one_hot_group_period(*_group_period(data.z))
    data.rel_vec = _rel_vectors(data.pos, data.edge_index).type(torch.float64)
    data.abs_distances = data.rel_vec.norm(dim=1).type(torch.float64)
    data.rel_vec = torch.nn.functional.normalize(data.rel_vec, p=2, dim=-1)
    data.y = data.y[:, 0].type(torch.float64)
    return data


def tpnn_transform_qm9_mpnn(data):
    def _rel_vectors(coordinates, bonds):
        origin_pos = coordinates[bonds[0]]
        neighbor_pos = coordinates[bonds[1]]
        return neighbor_pos - origin_pos

    data = data.clone()
    data.x = data.x.type(torch.float64)
    data.rel_vec = _rel_vectors(data.pos, data.edge_index).type(torch.float64)
    data.abs_distances = data.rel_vec.norm(dim=1).type(torch.float64)
    data.rel_vec = torch.nn.functional.normalize(data.rel_vec, p=2, dim=-1)
    data.y = data.y[:, 0].type(torch.float64)
    return data


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
        loss = F.mse_loss(logits, normalizer.tfm(targets))  # mse_loss # l1_loss
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
        loss = F.mse_loss(logits, normalizer.tfm(targets))  # mse_loss # l1_loss

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
        "target": "y",  # dipole moment
        "target_norm": [2.6822, 1.4974],  # mean, std
        "dataset_split_path": os.path.join(datasets_dir, "QM9", "randsplit_qm9.npy"),
        "b_size": 16,  # 64,

        "dataset": torch_geometric.datasets.QM9,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "QM9"),
            "transform": tpnn_transform_qm9
        },

        "model": LambdaZero.models.TPNN_v2,
        "model_config": {},
        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.0001  # 0.001
        },

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },

    "summaries_dir": summaries_dir,
    "memory": 8 * 10 ** 9,  # 20 * 10 ** 9,

    "stop": {"training_iteration": 100},
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
