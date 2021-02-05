import time, random, sys
import os.path as osp
import numpy as np
import torch
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
from LambdaZero.examples.bayesian_models.bayes_tune.brr import BRR

from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_epoch,eval_epoch, train_mcdrop, \
    mcdrop_mean_variance
from LambdaZero.examples.bayesian_models.bayes_tune import aq_config
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


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

    n1 = int(all_targets.shape[0] * 0.01)
    frac_top1percent = np.asarray(train_sorted[:n1] <= all_sorted[n1],dtype=np.float).mean()
    n10 = int(all_targets.shape[0] * 0.1)
    frac_top10percent = np.asarray(train_sorted[:n10] <= all_sorted[n10], dtype=np.float).mean()
    n25 = int(all_targets.shape[0] * 0.25)
    frac_top25percent = np.asarray(train_sorted[:n25] <= all_sorted[n25], dtype=np.float).mean()

    return {"aq_top15_regret":top15_regret, "aq_top50_regret":top50_regret, "aq_top15":aq_top15, "aq_top50":aq_top50,
            "aq_frac_top1_percent":frac_top1percent, "aq_frac_top10_percent":frac_top10percent,
            "aq_frac_top25_percent":frac_top25percent}


class UCB(tune.Trainable):
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
        self.regressor = self.config["regressor"](**config["regressor_config"])
        self.regressor.fit(self.train_loader,self.val_loader)

    def _train(self):
        idxs = self.acquire_batch()
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
        # print([len(s) for s in scores])
        # compute acquisition metrics
        _scores = aq_regret(self.train_loader, self.ul_loader, self.config["regressor_config"]["config"])
        scores = {**scores, **_scores}
        return scores

    def acquire_batch(self):
        # fixme ! - not sure how to deal with passing around train loade here ?!
        mean, var = self.regressor.get_mean_variance(self.ul_loader, self.train_loader)
        if self.config["minimize_objective"]:
            scores = -mean + (self.config["kappa"] * var)
        else:
            scores = mean + (self.config["kappa"] * var)
        # noise is not a part of original UCT but is added here
        noise = self.config["epsilon"] * np.random.uniform(size=mean.shape[0])
        scores = scores + noise
        idxs = np.argsort(-scores)[:self.config["aq_size"]]
        #print(mean, self.config["kappa"],self.config["epsilon"], noise)
        #print(np.mean(scores), np.mean(scores[idxs]))
        return idxs


DEFAULT_CONFIG = {
    "acquirer_config": {
        "run_or_experiment": UCB,
        "config":{
            "data": aq_config.data_config,
            "regressor": MCDrop,
            "regressor_config": aq_config.regressor_config,
            "aq_size0": 1000,
            "aq_size": 100,
            "kappa": 0.2,
            "epsilon": 0.0,
            "minimize_objective":True,
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 10},
        "resources_per_trial": {"cpu": 4, "gpu": 1.0}
    },
    "memory": 10 * 10 ** 9
}




if __name__ == "__main__":
    if len(sys.argv) >= 2: config_name = sys.argv[1]
    else: config_name = "ucb015_deup"
    config = getattr(aq_config, config_name)
    config = merge_dicts(DEFAULT_CONFIG, config)
    config["acquirer_config"]["name"] = config_name
    ray.init(_memory=config["memory"])

    print(config)
    tune.run(**config["acquirer_config"])

    #tune.run(**brr_config["acquirer_config"])