import torch
import torch_geometric

import os
import sys
from functools import partial

import ray
from ray import tune

from LambdaZero.utils import get_external_dirs, RegressorWithSchedulerOnBatch, RegressorWithSchedulerOnEpoch
from LambdaZero.utils import train_epoch, eval_epoch
import LambdaZero.models

from LambdaZero.datasets.utils import graph_add_directions_and_distances, graph_add_rsqrt_degree_norm, graph_select_targets

from LambdaZero.examples.tpnn import configs_representation
from LambdaZero.examples.tpnn import configs_radial_model
from LambdaZero.examples.tpnn import configs_gate
from LambdaZero.examples.tpnn import configs_optimizer
from LambdaZero.examples.tpnn import configs_scheduler


datasets_dir, _, summaries_dir = get_external_dirs()
targets_list = [0]  # dipole moment

assert len(sys.argv) == 7, "python3 train_tpnn_qm9.py representation radial_config gate_config optimizer_config scheduler_config r_cut"

_, representation_version, radial_model_version, gate_version, optimizer_version, scheduler_version, r_cut = sys.argv
representation = getattr(configs_representation, representation_version)
radial_model = getattr(configs_radial_model, radial_model_version)
gate = getattr(configs_gate, gate_version)
optimizer = getattr(configs_optimizer, optimizer_version)
scheduler = getattr(configs_scheduler, scheduler_version)
r_cut = float(r_cut)

pre_transform = torch_geometric.transforms.Compose([
    torch_geometric.transforms.RadiusGraph(r=r_cut, loop=False, max_num_neighbors=64, flow='target_to_source'),
    graph_add_rsqrt_degree_norm
])
transform = torch_geometric.transforms.Compose([
    graph_add_directions_and_distances,
    partial(graph_select_targets, target_idx=torch.tensor(targets_list, dtype=torch.int64))
])


config = {
    "trainer": RegressorWithSchedulerOnEpoch,
    "trainer_config": {
        "dtype": torch.float64,
        "target": "y",
        "target_norm": [2.68, 1.5],  # mean, std
        "dataset_split_path": os.path.join(datasets_dir, "QM9", "randsplit_qm9_110_10_10_v0.npy"),
        "batch_size": 16,  # 64,

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
            "radial_model": radial_model,
            "gate": gate,
            "pooling": 'set2set',
            "fc": True
        },
        "optimizer": optimizer,
        "scheduler": scheduler,

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
              f"-{config['trainer_config']['model_config']['radial_model']['type'].__name__}_{radial_model_version}" \
              f"-gate_{gate_version}" \
              f"-aggr_{config['trainer_config']['model_config']['pooling']}" \
              f"-fc_{config['trainer_config']['model_config']['fc']}" \
              f"-optim_{config['trainer_config']['optimizer']['type'].__name__}_{optimizer_version}" \
              f"-{config['trainer_config']['scheduler']['type'].__name__}_{scheduler_version}"


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
