import os.path as osp
import numpy as np
from torch_geometric import transforms as T
import LambdaZero.inputs
import LambdaZero.utils
from LambdaZero.examples.bayesian_models.bayes_tune.functions import *
import ray
from ray import tune

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


config = {
        "data":{
        "target": "gridscore",
        "dataset_creator": LambdaZero.inputs.dataset_creator_v1,
        "dataset_split_path": osp.join(datasets_dir,
                                       "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
        #"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore", "smi"],
            "transform": T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()]),
            "file_names":
                ["Zinc15_2k"],
            #["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],

        },
        "b_size": 40,
        "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057]),
    }
}


def sample_x(loader):
    fps = np.stack([d.fp for d in loader.dataset], axis=0)
    return fps

class BRR(tune.Trainable):
    def _setup(self, config):
        self.config = config

    def fit(self, train_loader, val_loader):
        train_targets_norm = sample_targets(train_loader, self.config)
        val_targets_norm = sample_targets(val_loader, self.config)
        train_fps, val_fps = sample_x(train_loader), sample_x(val_loader)
        scores, self.clf = bayesian_ridge(train_fps, val_fps, train_targets_norm, val_targets_norm, self.config)
        return [scores]

    def get_mean_variance(self, loader):
        mean, std = self.clf.predict(sample_x(loader), return_std=True)
        return mean, std



if __name__ == "__main__":
    ray.init()
    brr = BRR(config)
    train_loader, val_loader = config["data"]["dataset_creator"](config["data"])
    scores = brr.fit(train_loader, val_loader)
    print(scores)
    print(brr.get_mean_variance(val_loader))