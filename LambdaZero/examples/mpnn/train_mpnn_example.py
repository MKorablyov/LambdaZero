import os
from functools import partial

import numpy as np
import torch
import torch_geometric

from torch.nn.functional import mse_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error

import ray
from ray import tune

from LambdaZero.utils import get_external_dirs
from LambdaZero.utils import Trainer, StandardScaler
from LambdaZero.utils import train_epoch, val_epoch
from LambdaZero.models import MPNNet
from LambdaZero.datasets.docked import DockedDataset

from LambdaZero.chem import mpnn_feat
from rdkit import Chem


def regret(y_true, y_pred, n):
    ranked_y_true = y_true[np.argsort(y_true)]
    pred_ranked_y_true = y_true[np.argsort(y_pred)]
    return np.median(pred_ranked_y_true[:n]) - np.median(ranked_y_true[:n])


regret_top15 = partial(regret, n=15)
regret_top50 = partial(regret, n=50)


def example_featurization(graph):
    mol = Chem.MolFromSmiles(graph.smiles)
    atom_features, _, bonds, bond_features = mpnn_feat(mol, ifcoord=False)
    graph.x = torch.tensor(atom_features, dtype=torch.float32)
    graph.edge_index = torch.tensor(np.concatenate([bonds.T, np.flipud(bonds.T)], axis=1), dtype=torch.int64)
    graph.edge_attr = torch.tensor(np.concatenate([bond_features, bond_features], axis=0), dtype=torch.float32)
    return graph


datasets_dir, _, summaries_dir = get_external_dirs()

COMMON_PRE_TRANSFORM = torch_geometric.transforms.Compose(
    [DockedDataset.get_best_conformation_minimal,
     example_featurization]
)

COMMON_METRICS = [
    ('mse', mean_squared_error, 'y'),
    ('mae', mean_absolute_error, 'y'),
    ('regret_top15', regret_top15, 'y'),
    ('regret_top50', regret_top50, 'y')
]

config_name = "example"
config = {
    "trainer": Trainer,
    "trainer_config": {
        "dtype": torch.float32,
        "device": torch.device('cuda'),

        "train_dataset": {
            # There are two different kinds of aliases.
            # The one directly below acts as a part of name for loss/metrics: train_{alias}_{metric_alias}.
            # Alias in config for DockedDataset allows to have multiple processed files for the same raw file by creating subdirectory named after alias.
            # Those two aliases can be different.
            "alias": 'zinc',
            # batch size for train and each of the validation sets can be different
            # when batch size for validation set is not specified - batch size of train set acts as a default
            "batch_size": 16,
            "loss_function": mse_loss,
            "target": 'y',
            "class": DockedDataset,
            "config": {
                "root": os.path.join(datasets_dir, "ZINC20"),
                "file_name": 'zinc20_sample_1.pth',
                "alias": 'best_conf',
                "pre_transform": COMMON_PRE_TRANSFORM
            },
            # (optional) idx_path should point to npy file with single array of ordinal numbers of entries to be used
            # If it is not provided all entries are gonna be used, "idx_path": None can be omitted entirely
            "idx_path": None,
            # scaling can be omitted entirely as well, in that case default IdentityScaler is supplied for compatibility
            # class supplied here should be inherited from Scaler base class
            "scaling": {
                "class": StandardScaler,
                "config": {
                    "mean": -8.6,
                    "std": 1.1
                }
            },
            # Should be a list of tuples: (alias, func, target)
            # Note: metrics operate on target in its original scale, NOT normalized one
            "metrics": COMMON_METRICS
        },

        "validation_datasets": [
            {
                "alias": 'zinc_sample_2',
                "batch_size": 32,
                "class": DockedDataset,
                "config": {
                    "root": os.path.join(datasets_dir, "ZINC20"),
                    "file_name": 'zinc20_sample_2.pth',
                    "alias": 'best_conf',
                    "pre_transform": COMMON_PRE_TRANSFORM
                },
                "metrics": COMMON_METRICS
            },
            {
                "alias": 'zinc_sample_3',
                "batch_size": 48,
                "class": DockedDataset,
                "config": {
                    "root": os.path.join(datasets_dir, "ZINC20"),
                    "file_name": 'zinc20_sample_3.pth',
                    "alias": 'best_conf',
                    "pre_transform": COMMON_PRE_TRANSFORM
                },
                "idx_path": os.path.join(datasets_dir, "ZINC20", "val_idx_zinc20_sample_3.npy"),
                "metrics": COMMON_METRICS
            }
        ],

        "model": {
            "class": MPNNet,
            "config": {}
        },

        "optimizer": {
            "class": torch.optim.Adam,
            "config": {
                "lr": 1e-3
            }
        },

        "scheduler": {
            "class": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "config": {
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-6
            },
            # Scheduler step can be called either after each 'epoch', or after each 'batch'
            "trigger": 'epoch',
            # Some schedulers expect validation metric as an input on each step
            # specify as string of form '{validation_dataset_alias}_{metric_alias}'
            "target": 'zinc_sample_3_mae'
        },
        "train_epoch": train_epoch,
        "eval_epoch": val_epoch
    },

    "summaries_dir": summaries_dir,
    "memory": 8 * 10 ** 9,
    "object_store_memory": 2.5 * 10 ** 9,

    "stop": {"training_iteration": 200},
    "resources_per_trial": {
        "cpu": 4,
        "gpu": 1.0
    },
    "keep_checkpoint_num": 2,
    "checkpoint_score_attr": 'train_zinc_loss',
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 25
}


if __name__ == "__main__":
    ray.init(local_mode=True, _memory=config["memory"], object_store_memory=config["object_store_memory"])

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"],
                        name=config_name)
