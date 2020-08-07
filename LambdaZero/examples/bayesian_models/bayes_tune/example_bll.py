import os.path as osp
import numpy as np
from torch_geometric import transforms as T
import LambdaZero.inputs
import LambdaZero.utils
from LambdaZero.examples.bayesian_models.bayes_tune.functions import bayesian_ridge

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
transform = T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()])


config = {
        "target": "gridscore",
        "dataset_creator": LambdaZero.utils.dataset_creator_v1,
        "dataset_split_path": osp.join(datasets_dir,
                                       "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
        # "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore", "smi"],
            "transform": transform,
            "file_names":
                ["Zinc15_2k"],
            # ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],

        },
        "b_size": 40,
        "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057]),
    }



def bll_on_fps(config):
    "this just computes uncertainty on FPs"
    # make dataset
    train_loader, val_loader = config["dataset_creator"](config)
    train_targets = np.concatenate([getattr(d, config["target"]).cpu().numpy() for d in train_loader.dataset])
    train_fps = np.stack([d.fp for d in train_loader.dataset], axis=0)
    val_targets = np.concatenate([getattr(d, config["target"]).cpu().numpy() for d in val_loader.dataset])
    val_fps = np.stack([d.fp for d in val_loader.dataset], axis=0)
    train_targets_norm = config["normalizer"].tfm(train_targets)
    val_targets_norm = config["normalizer"].tfm(val_targets)
    scores = bayesian_ridge(train_fps ,val_fps ,train_targets_norm, val_targets_norm, config)
    print(scores)

bll_on_fps(config)