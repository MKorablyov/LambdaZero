import torch
import os.path as osp
import LambdaZero.inputs
from LambdaZero.examples.egnn.egnn import EGNNet
from LambdaZero.utils import get_external_dirs, BasicRegressor, RegressorWithSchedulerOnEpoch, train_epoch, eval_epoch

datasets_dir, programs_dir, summaries_dir = get_external_dirs()
transform = LambdaZero.utils.Complete()

# summary: more layers+scheduler doesn't help, 1e-3 better but fluctuates; AdamW 5e-4 > Adam 5e-4, but 1e-3 is similar
# amsgrad doesn't help with Adam(W)
# CosineAnnealingLR/CosineAnnealingWarmRestarts scheduler helps with Adam(W) + appropraite Tmax (85~100>125>250~500)
# ReduceLROnPlateau, CyclicLR, or OneCycleLR does not help
# egnn_001_007_30k or egnn_001_005 are the best

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "target": "dockscore",
        "target_norm": [-8.6597, 1.0649],
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
        "batch_size": 96,
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": osp.join(datasets_dir, "brutal_dock/seh"),
            "props": ["dockscore", "smiles"],
            "transform": transform,
            "file_names": ["Zinc20_docked_neg_randperm_3k"],
        },
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,

        "model": EGNNet,
        "model_config": {
            "n_layers": 3,
            "feats_dim": 14,
            "pos_dim": 3,
            "edge_attr_dim": 0,
            "m_dim": 128,
            "control_exp": False,
            "settoset": False,
        },
    },
    "summaries_dir": summaries_dir,
    "memory": 50 * 10 ** 9,
    "object_store_memory": 50 * 10 ** 9,

    "stop": {"training_iteration": 500},
    "resources_per_trial": {
        "cpu": 8,
        "gpu": 2.0
    },
    "keep_checkpoint_num": 1,
    "checkpoint_score_attr": "train_loss",
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 100,
}

egnn_000 = {
    "trainer_config": {
        "batch_size": 96,
        "optimizer": {
            "type": torch.optim.Adam,
            "config": {
                "weight_decay": 1e-16,
                "lr": 5e-4,
            },
        },
    },
}


egnn_000_001 = { # a little bit better than 000
    "trainer_config": {
        "batch_size": 96,
        "optimizer": {
            "type": torch.optim.AdamW,
            "config": {
                "lr": 5e-4,
                "weight_decay": 1e-3,
                # amsgrad doesn't help!
            },
        },
    },
}

egnn_000_002 = { # a little better than 000_001, but fluctuates more
    "trainer_config": {
        "batch_size": 96,
        "optimizer": {
            "type": torch.optim.AdamW,
            "config": {
                "lr": 1e-3,
                "weight_decay": 5e-3,
                # amsgrad doesn't help!
            },
        },
    },
}

egnn_000_003 = { # does as well as 000_002
    "trainer_config": {
        "batch_size": 96,
        "optimizer": {
            "type": torch.optim.Adam,
            "config": {
                "lr": 1e-3,
                # amsgrad doesn't help
            },
        },
    },
}

egnn_001_005 = { # Best so far for small dataset
    "trainer": RegressorWithSchedulerOnEpoch,
    "trainer_config": {
        "batch_size": 96,
        "optimizer": {
            "type": torch.optim.AdamW,
            "config": {
                "lr": 1e-3, # better than 5e-4 here
            },
        },
        "scheduler": {
            "type": torch.optim.lr_scheduler.CosineAnnealingLR,
            "config": {
                "T_max": 100,
            },
        },
    },
}


egnn_001_007_30k = { # Best so far for larger dataset
    "trainer": RegressorWithSchedulerOnEpoch,
    "trainer_config": {
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k.npy"),
        "dataset_config": {
            "root": osp.join(datasets_dir, "brutal_dock/seh"),
            "props": ["dockscore", "smiles"],
            "transform": transform,
            "file_names": ["Zinc20_docked_neg_randperm_30k"],
        },
        "batch_size": 96,
        "optimizer": {
            "type": torch.optim.AdamW,
            "config": {
                "lr": 1e-3, # better than 5e-4
            },
        },
        "scheduler": {
            "type": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "config": {
                "T_0": 100, # 90, 250 is worse
            },
        },
    },
}

egnn_001_011_30k = {
    "trainer": RegressorWithSchedulerOnEpoch,
    "trainer_config": {
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k.npy"),
        "dataset_config": {
            "root": osp.join(datasets_dir, "brutal_dock/seh"),
            "props": ["dockscore", "smiles"],
            "transform": transform,
            "file_names": ["Zinc20_docked_neg_randperm_30k"],
        },
        "optimizer": {
            "type": torch.optim.AdamW,
            "config": {
                "lr": 1e-3, # better than 5e-4
            },
        },
        "scheduler": {
            "type": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "config": {
                "T_0": 85, # 90, 250 is worse
            },
        },
    },
}

