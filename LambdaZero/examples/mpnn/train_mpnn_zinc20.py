import torch
import torch_geometric

import os
import sys

import ray
from ray import tune

from LambdaZero.datasets.zinc20.dataset import ZINC20

from LambdaZero.utils import get_external_dirs, RegressorWithSchedulerOnBatch, RegressorWithSchedulerOnEpoch
from LambdaZero.utils import train_epoch, eval_epoch

from LambdaZero.datasets.utils import graph_add_distances, mol_graph_add_bond_type, mol_graph_add_junction_tree

from LambdaZero.examples.mpnn import configs_model
from LambdaZero.examples.mpnn import configs_optimizer
from LambdaZero.examples.mpnn import configs_scheduler


datasets_dir, _, summaries_dir = get_external_dirs()

assert len(sys.argv) == 4, "python3 train_mpnn_zinc20.py model_config optimizer_config scheduler_config"

_, model_version, optimizer_version, scheduler_version = sys.argv
model = getattr(configs_model, model_version)
optimizer = getattr(configs_optimizer, optimizer_version)
scheduler = getattr(configs_scheduler, scheduler_version)


def graph_select_best_conf(graph):
    graph.pos = graph.pos_docked[0]
    graph.y = graph.dockscores[0].type(torch.float32)
    graph.z = graph.z.type(torch.int64)
    del graph.pos_docked  # otherwise batch collate function would fail
    return graph


transform = torch_geometric.transforms.Compose([
    graph_select_best_conf,
    graph_add_distances,
    mol_graph_add_bond_type,
    mol_graph_add_junction_tree
])


config = {
    "trainer": RegressorWithSchedulerOnEpoch,
    "trainer_config": {
        "dtype": torch.float32,
        "target": "y",
        "target_norm": [-8.66, 1.1],  # mean, std over best(!) conformations on the whole(!) ZINC20 set
        "dataset_split_path": os.path.join(datasets_dir, "zinc20/raw/split_zinc20_graphs_100k.npy"),
        "batch_size": 16,  # 64,

        "dataset": ZINC20,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "zinc20"),
            "transform": transform,
            "file_name": "zinc20_graphs_100k"
        },
        **model,
        "optimizer": optimizer,
        "scheduler": scheduler,

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    },

    "summaries_dir": summaries_dir,
    "memory": 8 * 10 ** 9,  # 20 * 10 ** 9,
    "object_store_memory": 2.5 * 10 ** 9,

    "stop": {"training_iteration": 200},
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

config_name = f"{config['trainer_config']['model'].__name__}" \
              f"-optim_{config['trainer_config']['optimizer']['type'].__name__}_{optimizer_version}" \
              f"-{config['trainer_config']['scheduler']['type'].__name__}_{scheduler_version}"


if __name__ == "__main__":
    ray.init(_memory=config["memory"], object_store_memory=config["object_store_memory"])

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"],
                        name=config_name)
