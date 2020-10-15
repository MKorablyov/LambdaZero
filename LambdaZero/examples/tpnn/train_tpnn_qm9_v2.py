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
from LambdaZero.examples.tpnn import config_qm9 as config

import torch_geometric
from torch_geometric.utils import degree

from functools import partial

datasets_dir, _, summaries_dir = get_external_dirs()
targets_list = [0]  # dipole moment


def add_norm(graph):
    origin_nodes, _ = graph.edge_index  # origin, neighbor
    node_degrees = degree(origin_nodes, num_nodes=graph.x.size(0))
    graph.norm = node_degrees[origin_nodes].type(torch.float64).rsqrt()  # 1 / sqrt(degree(i))
    return graph


def add_directions_and_distances(graph):
    # direction
    origin_pos = graph.pos[graph.edge_index[0]]
    neighbor_pos = graph.pos[graph.edge_index[1]]
    rel_vec = (neighbor_pos - origin_pos).type(torch.float64)
    graph.rel_vec = torch.nn.functional.normalize(rel_vec, p=2, dim=-1)
    # distance
    graph.abs_distances = rel_vec.norm(dim=1)
    return graph


def select_qm9_targets(graph, target_idx):
    graph.y = graph.y[0, target_idx].type(torch.float64)
    return graph


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
        preds_normalized = model(data).view(-1)
        loss = F.mse_loss(preds_normalized, normalizer.tfm(targets))  # mse_loss # l1_loss
        loss.backward()
        optimizer.step()

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.itfm(preds_normalized).detach().cpu().numpy())

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

        preds_normalized = model(data).view(-1)
        loss = F.mse_loss(preds_normalized, normalizer.tfm(targets))  # mse_loss # l1_loss

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.itfm(preds_normalized).detach().cpu().numpy())

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


assert len(sys.argv) == 3, "Execution should be in the form python3 train_tpnn_qm9_v2.py model_config_name r_cut"

config_name = sys.argv[1]
r_cut = float(sys.argv[2])
config = getattr(config, config_name)

pre_transform = torch_geometric.transforms.Compose([
    torch_geometric.transforms.RadiusGraph(r=r_cut, loop=False, max_num_neighbors=64, flow='target_to_source'),
    add_norm
])
transform = torch_geometric.transforms.Compose([
    add_directions_and_distances,
    partial(select_qm9_targets, target_idx=torch.tensor(targets_list, dtype=torch.int64))
])


TPNN_CONFIG = {
    "trainer": TPNNRegressor,
    "trainer_config": {
        "target": "y",
        "target_norm": [2.68, 1.5],  # mean, std
        "dataset_split_path": os.path.join(datasets_dir, "QM9", "randsplit_qm9.npy"),
        "b_size": 16,  # 64,

        "dataset": torch_geometric.datasets.QM9,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "QM9"),
            "pre_transform": pre_transform,
            "transform": transform
        },

        "model": LambdaZero.models.TPNN_ResNet_Avg,
        "model_config": {
            "max_z": 10,
            "avg_n_atoms": 18.025
        },
        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.0001
        },

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },

    "summaries_dir": summaries_dir,
    "memory": 8 * 10 ** 9,  # 20 * 10 ** 9,

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
