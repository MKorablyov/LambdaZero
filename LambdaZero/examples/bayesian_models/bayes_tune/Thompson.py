import time, random, sys
import os.path as osp
import numpy as np
import torch
from torch import distributions
import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from torch_geometric.data import DataLoader
from torch_geometric import transforms as T
from torch.utils.data import Subset, ConcatDataset
import LambdaZero.utils
import LambdaZero.models
import LambdaZero.inputs
from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop
# from LambdaZero.models import MPNNetDrop
from LambdaZero.examples.bayesian_models.bayes_tune.brr import BRR

from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_epoch,eval_epoch, train_mcdrop, mcdrop_mean_variance, train_mpnn_brr, mpnn_brr_mean_variance
from LambdaZero.examples.bayesian_models.bayes_tune import aq_config

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def aq_regret(train_loader, ul_loader, config):
    train_targets = np.concatenate([getattr(d, config["data"]["target"]).cpu().numpy() for d in train_loader.dataset],0)
    ul_targets = np.concatenate([getattr(d, config["data"]["target"]).cpu().numpy() for d in ul_loader.dataset],0)
    all_targets = np.concatenate([train_targets, ul_targets],0)
    train_sorted = train_targets[np.argsort(train_targets)]
    all_sorted = all_targets[np.argsort(all_targets)]

    top15_regret = np.median(train_sorted[:15]) - np.median(all_sorted[:15])
    top50_regret = np.median(train_sorted[:50]) - np.median(all_sorted[:50])
    aq_top15 = np.median(train_sorted[:15])
    aq_top50 = np.median(train_sorted[:15])

    n = int(all_targets.shape[0] * 0.01)
    frac_top1percent = np.asarray(train_sorted[:n] <= all_sorted[n],dtype=np.float).mean()
    return {"aq_top15_regret":top15_regret, "aq_top50_regret":top50_regret, "aq_top15":aq_top15, 
"aq_top50":aq_top50,
            "aq_frac_top1_percent":frac_top1percent}


class Thompson(tune.Trainable):
    def _setup(self, config):
        self.config = config
        # load dataset
        self.dataset = config["data"]["dataset"](**config["data"]["dataset_config"])
        ul_idxs, val_idxs, test_idxs = np.load(self.config["data"]["dataset_split_path"], allow_pickle=True)
        
        np.random.shuffle(ul_idxs) # randomly acquire batch zero
        train_idxs = ul_idxs[:self.config["aq_size0"]]
        ul_idxs = ul_idxs[self.config["aq_size0"]:]
        train_set = Subset(self.dataset, train_idxs.tolist())
        ul_set = Subset(self.dataset, ul_idxs.tolist())
        val_set = Subset(self.dataset, val_idxs.tolist())
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=config["data"]["b_size"])
        self.ul_loader = DataLoader(ul_set, batch_size=config["data"]["b_size"])
        self.val_loader = DataLoader(val_set, batch_size=config["data"]["b_size"])

        # make model with uncertainty
        model = LambdaZero.models.MPNNetDrop
        self.config['model'] = model
        config['regressor_config']['config']['train'] = train_mpnn_brr
        config['regressor_config']['config']['get_mean_variance'] = mpnn_brr_mean_variance
        self.regressor = self.config["regressor"](**config["regressor_config"])
        
        self.regressor.fit(self.train_loader,self.val_loader)

    def _train(self):
        # for _ in range(self.config['data']['b_size']):
        idxs = self.acquire_batch(batch_size = self.config['data']['b_size'])
        scores = self.update_with_seen(idxs)
        return scores

    def update_with_seen(self, idxs):
        # update train/unlabeled datasets
        aq_mask = np.zeros(len(self.ul_loader.dataset.indices),dtype=np.bool)
        aq_mask[idxs] = True
        aq_idxs = np.asarray(self.ul_loader.dataset.indices)[aq_mask].tolist()
        ul_idxs = np.asarray(self.ul_loader.dataset.indices)[~aq_mask].tolist()
        train_idxs = self.train_loader.dataset.indices + aq_idxs
        train_set = Subset(self.dataset, train_idxs)
        ul_set = Subset(self.dataset, ul_idxs)
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config["data"]["b_size"])
        self.ul_loader = DataLoader(ul_set, batch_size=self.config["data"]["b_size"])
        # fit model to the data

        scores = self.regressor.fit(self.train_loader, self.val_loader)[-1]

        # compute acquisition metrics
        _scores = aq_regret(self.train_loader, self.ul_loader, self.config["regressor_config"]["config"])
        scores = {**scores, **_scores}
        return scores

    def acquire_batch(self, batch_size=500):
        idxs = []
        
        for _ in range(batch_size):
            embs = []
            y = np.concatenate([getattr(d, 'gridscore').cpu().numpy() for d in self.train_loader.dataset],0)
            for bidx, data in enumerate(self.train_loader):
                data = data.to(device)
                emb = self.regressor.model.get_embed(data, False)
                embs.append(emb.detach().cpu().numpy())
            embs = np.concatenate(embs, 0)

            ul_embs = []

            for bidx, data in enumerate(self.ul_loader):
                data = data.to(device)
                emb = self.regressor.model.get_embed(data, False)
                ul_embs.append(emb.detach().cpu().numpy())
            ul_embs = np.concatenate(ul_embs, 0)
            idx = self.get_posterior_sample(X_in = ul_embs,X_trn = embs, y = y, idxs = idxs, X_feature_size = 64, sigma_y = 0.2)
            if idx: idxs.append(int(idx))
        return idxs

    def get_posterior_sample(self, X_in, X_trn, y, idxs, X_feature_size,sigma_y):
        # Murphy 7.55
        w_0 = np.zeros(X_feature_size)
        V_0 = np.diag([sigma_y] * X_feature_size)**2
        V0_inv = np.linalg.inv(V_0)
        V_n = sigma_y**2 * np.linalg.inv(sigma_y**2 * V0_inv + (X_trn.T @ X_trn))
        w_n = V_n @ V0_inv @ w_0 + 1 / (sigma_y**2) * V_n @ X_trn.T @ y

        #sample w from distribution N(w_n,V_n)
        samples = np.random.multivariate_normal(w_n, V_n, size = 1)
        #pick the argmax of expectation of y* which is x*w
        y_star = X_in @ samples.T
        #add best the train set
        y_star = y_star.argsort()
        for i in y_star:
            if i not in idxs:
                return i

DEFAULT_CONFIG = {
    "acquirer_config": {
        "run_or_experiment": Thompson,
        "config":{
            "data": aq_config.data_config,
            "regressor": MCDrop,
            "regressor_config": aq_config.regressor_config,
            "aq_size0": 200,
            "aq_size": 50,
            "minimize_objective":True,
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 20},
        "resources_per_trial": {"cpu": 4, "gpu": 1.0}
    },
    "memory": 10 * 10 ** 9
}

if __name__ == "__main__":
    if len(sys.argv) >= 2: config_name = sys.argv[1]
    else: config_name = "ts005_testing"
    config = getattr(aq_config, config_name)
    config = merge_dicts(DEFAULT_CONFIG, config)
    config["acquirer_config"]["name"] = config_name
    ray.init(memory=config["memory"])

    print(config)
    tune.run(**config["acquirer_config"])
