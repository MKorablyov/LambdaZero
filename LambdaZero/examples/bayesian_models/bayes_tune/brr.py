import os.path as osp
import numpy as np
from torch.utils.data import Dataset,DataLoader,Subset
from torch_geometric import transforms as T
import ray
from ray import tune
from LambdaZero.inputs import random_split
import LambdaZero.inputs
import LambdaZero.utils
from LambdaZero.examples.bayesian_models.bayes_tune.functions import *
# from LambdaZero.examples.drug_comb.new_drugcomb_data_v2 import DrugCombEdge

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

def fp_feat(loader):
    fps = np.stack([d.fp for d in loader.dataset], axis=0)
    return fps

class BRR(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.get_feat = self.config["regressor_config"]["get_feat"]
        # self.feature_dim = 64

    def fit(self, train_loader, val_loader):
        train_feat, val_feat = self.get_feat(train_loader), self.get_feat(val_loader)
        train_targets_norm = sample_targets(train_loader, self.config)
        val_targets_norm = sample_targets(val_loader, self.config)
        scores, self.clf = bayesian_ridge(train_feat, val_feat, train_targets_norm, val_targets_norm, self.config)

        # self.alpha = self.clf.lambda_
        # self.beta = self.clf.alpha_

        # self.K = self.beta * train_feat.T @ train_feat + self.alpha * torch.eye(self.feature_dim)   # [M, M]
        # self.chol_K = torch.cholesky(self.K)

        # projected_y = x_train.T @ train_targets_norm
        # k_inv_projected_y = torch.cholesky_solve(projected_y, self.chol_K)
        # self.m = self.beta * k_inv_projected_y  # [M, 1]

        return [scores]

    def get_mean_variance(self, loader):
        mean, std = self.clf.predict(self.get_feat(loader), return_std=True)
        return mean, std


DEFAULT_CONFIG = {
    "regressor_config":{"get_feat":fp_feat},
    "data":{
        "target": "gridscore",
        "dataset_creator": LambdaZero.inputs.dataset_creator_v1,
        "dataset_split_path": osp.join(datasets_dir,
                                       "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
        # "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore", "smi"],
            "transform": T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()]),
            "file_names": ["Zinc15_2k"],
            #["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
        },
        "b_size": 40,
        "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057]),
    }
}





if __name__ == "__main__":
    config = DEFAULT_CONFIG
    ray.init(memory=5*10**9)
    brr = BRR(config)
    train_loader, val_loader = config["data"]["dataset_creator"](config["data"])
    scores = brr.fit(train_loader, val_loader)

    print(brr.get_mean_variance(val_loader))

    # config = config_drugcomb
    # dataset = DrugCombEdge()
    # idxs = np.where(dataset.data.ddi_edge_classes.numpy() == 1700)[0]
    #
    # dataset_cell = Subset(dataset,idxs)
    # train_idx, val_idx, test_ixs = random_split(len(dataset_cell), [0.8, 0.1, 0.1])
    #
    #
    # train_set = Subset(dataset_cell, train_idx)
    # val_set = Subset(dataset_cell, val_idx)
    # brr = BRR(config)
    # print(brr.fit(train_set, val_set))