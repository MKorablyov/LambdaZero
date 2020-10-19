import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

import ray
from ray import tune
# from ray.rllib.utils import merge_dicts

from LambdaZero.utils import get_external_dirs, TPNNRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models

from LambdaZero.examples.tpnn import configs_representation
from LambdaZero.examples.tpnn import configs_radial_model
from LambdaZero.examples.tpnn import configs_gate
from LambdaZero.examples.tpnn import configs_optimizer
from LambdaZero.examples.tpnn import configs_scheduler

import torch_geometric
from torch_geometric.utils import degree

import e3nn.radial
import e3nn.point.gate

from functools import partial

datasets_dir, _, summaries_dir = get_external_dirs()
targets_list = [0]  # dipole moment


def graph_add_norm(graph):
    origin_nodes, _ = graph.edge_index  # origin, neighbor
    node_degrees = degree(origin_nodes, num_nodes=graph.x.size(0))
    graph.norm = node_degrees[origin_nodes].type(torch.float64).rsqrt()  # 1 / sqrt(degree(i))
    return graph


def graph_add_directions_and_distances(graph):
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
    def closure():
        optimizer.zero_grad()
        preds_normalized = model(data).view(-1)
        loss = F.mse_loss(preds_normalized, normalizer.tfm(targets))  # mse_loss # l1_loss
        loss.backward()
        # LBFGS has operations float(closure_output), so usual way of returning multiple outputs fail
        # also, apparently closure is called multiple times for LBFGS, so if appends to list occur here it leads to multiple copies
        nonlocal batch_preds
        batch_preds = normalizer.itfm(preds_normalized).detach().cpu().numpy()
        return loss

    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss": 0}
    epoch_targets = []
    batch_preds = None
    epoch_preds = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])
        batch_mean_loss = optimizer.step(closure).item()

        # log stuff
        metrics["loss"] += batch_mean_loss * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(batch_preds)

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


assert len(sys.argv) == 7, "python3 train_tpnn_qm9.py r_cut representation radial_config gate_config optimizer_config scheduler_config"

r_cut = float(sys.argv[1])
representation_version, radial_model_version, gate_version, optimizer_version, scheduler_version = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
representation = getattr(configs_representation, representation_version)
radial_model_config = getattr(configs_radial_model, radial_model_version)
gate_config = getattr(configs_gate, gate_version)
optimizer_config = getattr(configs_optimizer, optimizer_version)
scheduler_config = getattr(configs_scheduler, scheduler_version)

pre_transform = torch_geometric.transforms.Compose([
    torch_geometric.transforms.RadiusGraph(r=r_cut, loop=False, max_num_neighbors=64, flow='target_to_source'),
    graph_add_norm
])
transform = torch_geometric.transforms.Compose([
    graph_add_directions_and_distances,
    partial(select_qm9_targets, target_idx=torch.tensor(targets_list, dtype=torch.int64))
])


config = {
    "trainer": TPNNRegressor,
    "trainer_config": {
        "target": "y",
        "target_norm": [2.68, 1.5],  # mean, std
        "dataset_split_path": os.path.join(datasets_dir, "QM9", "randsplit_qm9_small.npy"),
        "b_size": 16,  # 64,

        "dataset": torch_geometric.datasets.QM9,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "QM9"),
            "pre_transform": pre_transform,
            "transform": transform
        },

        "model": LambdaZero.models.TPNN_v0,
        "model_config": {
            "max_z": 10,
            "representations": representation,
            "equivariant_model": LambdaZero.models.TPNN_ResNet,
            "radial_model": e3nn.radial.CosineBasisModel,
            "radial_model_config": radial_model_config,
            "gate": e3nn.point.gate.Gate,
            "gate_config": gate_config,
            "pooling": 'set2set',
            "fc": True
        },
        "optimizer": torch.optim.LBFGS,
        "optimizer_config": optimizer_config,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_config": scheduler_config,
        "scheduler_criteria": 'loss',  # on validation set

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },

    "summaries_dir": summaries_dir,
    "memory": 8 * 10 ** 9,  # 20 * 10 ** 9,

    "stop": {"training_iteration": 120},
    "resources_per_trial": {
        "cpu": 1,
        "gpu": 1.0
    },
    "keep_checkpoint_num": 2,
    "checkpoint_score_attr": "train_loss",
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 100000,  # 1;  0 is a default in tune.run
}

config_name = f"{config['trainer_config']['model_config']['equivariant_model'].__name__}" \
              f"-{r_cut}" \
              f"-repr_{representation_version}" \
              f"-{config['trainer_config']['model_config']['radial_model'].__name__}_{radial_model_version}" \
              f"-gate_{gate_version}" \
              f"-aggr_{config['trainer_config']['model_config']['pooling']}" \
              f"-fc_{config['trainer_config']['model_config']['fc']}" \
              f"-optim_{config['trainer_config']['optimizer'].__name__}_{optimizer_version}" \
              f"-{config['trainer_config']['scheduler'].__name__}_{scheduler_version}"


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
