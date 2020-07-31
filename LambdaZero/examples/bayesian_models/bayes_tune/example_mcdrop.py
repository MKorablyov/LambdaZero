import socket, os,sys, time, random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader

import ray
from ray import tune
from ray.rllib.utils import merge_dicts

from torch_geometric import transforms as T

from LambdaZero.utils import get_external_dirs, BasicRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.mpnn import config

from scipy.special import logsumexp
from matplotlib import pyplot as plt


transform = T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()])


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "mpnn000"
config = getattr(config, config_name)


def _compute_metrics(epoch_targets_norm, epoch_logits, normalizer):
    epoch_targets = normalizer.backward_transform(epoch_targets_norm)
    epoch_preds = normalizer.backward_transform(epoch_logits)
    metrics = {}
    #metrics["loss"] = metrics["loss"] / epoch_targets.shape[0] todo
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()
    # ranking
    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])
    metrics["epoch_targets_norm"] = epoch_targets
    metrics["epoch_preds"] = epoch_preds
    return metrics


def train_epoch(loader, model, optimizer, device, config):
    model.train()
    epoch_targets_norm = []
    epoch_logits = []

    N = len(loader)
    alpha = config['lengthscale'] ** 2 * (1 - config["drop_p"]) / (2. * N * config['tau'])

    fps = np.concatenate([d.fp for d in loader],axis=0)
    print("fingerprints", fps.shape)
    time.sleep(100)


    for bidx,data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        logits = model(data, do_dropout=True, drop_p=config["drop_p"] )
        targets_norm = config["normalizer"].forward_transform(targets)
        reg_loss = alpha * torch.stack([(p ** 2).sum() for p in model.parameters()]).sum()
        loss = F.mse_loss(logits, targets_norm) + reg_loss
        loss.backward()
        optimizer.step()

        epoch_targets_norm.append(targets_norm.detach().cpu().numpy())
        epoch_logits.append(logits.detach().cpu().numpy())

    epoch_targets_norm = np.concatenate(epoch_targets_norm,0)
    epoch_logits = np.concatenate(epoch_logits, 0)
    scores = _compute_metrics(epoch_targets_norm, epoch_logits, config["normalizer"])
    return scores


def eval_epoch(loader, model, device, config):
    model.eval()
    epoch_targets_norm = []
    epoch_logits = []
    for bidx, data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])
        logits = model(data, do_dropout=True, drop_p=config["drop_p"])
        targets_norm = config["normalizer"].forward_transform(targets)
        epoch_targets_norm.append(targets_norm.detach().cpu().numpy())
        epoch_logits.append(logits.detach().cpu().numpy())
    epoch_targets_norm = np.concatenate(epoch_targets_norm, 0)
    epoch_logits = np.concatenate(epoch_logits, 0)
    scores = _compute_metrics(epoch_targets_norm, epoch_logits, config["normalizer"])
    return scores

DEFAULT_CONFIG = {
    "regressor_config":{
        "run_or_experiment": BasicRegressor,
        "config": {
            "target": "gridscore",
            "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
                                           #"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
            "b_size": 50,
            "tau": 5.0,
            "T": 10,
            "drop_p": 0.1,
            "lengthscale": 1e-2,


            "dataset": LambdaZero.inputs.BrutalDock,
            "dataset_config": {
                "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                "props": ["gridscore", "smi"],
                "transform": transform,
                "file_names": ["Zinc15_2k"], #["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
            },
            "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057]),

            "model": LambdaZero.models.MPNNetDrop,
            "model_config": {},


            "optimizer": torch.optim.Adam,
            "optimizer_config": {
                "lr": 0.001
            },

            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 200},
        "resources_per_trial": {
            "cpu": 4,
            "gpu": 1.0
        },
        # "keep_checkpoint_num": 2,
        # "checkpoint_score_attr":"train_loss",
        # "num_samples":1,
        # "checkpoint_at_end": False,
    },
    "memory": 10 * 10 ** 9
}


# todo: dropout (p= 0.01, 0.03, 0.09, 0.5)
# todo: lengthscale (0.01, 0.03, ??? )
# dropout_hyperparameter (drop_layers=True, drop_mlp=True)
# todo: add BLL (from John's code)


def _log_lik(y, Yt_hat, tau):
    "computes log likelihood"
    # ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat) ** 2., 0)
    # - np.log(T)
    # - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau))

    ll = logsumexp(-0.5 * tau * (y[None] - Yt_hat) ** 2., 0)
    ll -= np.log(Yt_hat.shape[0])
    ll -= 0.5 * np.log(2 * np.pi)
    ll += 0.5 * np.log(tau)
    return ll


class MCDropRegressor(BasicRegressor):
    def __init__(self, config):
        super(BasicRegressor, self).__init__(config["config"])

    def fit(self, train_loader, val_loader):
        # update internal dataset
        self.train_loader, self.val_loader = train_loader, val_loader
        # make a new model
        self.model = self.config["model"](**self.config["model_config"])
        self.model.to(self.device)

        # todo allow ray native stopping
        all_scores = []
        for i in range(50):
            scores = self._train()
            if i % 25 == 0:
                y = [getattr(s, self.config["target"]).cpu().numpy() for s in self.val_set]
                y_norm = self.config["normalizer"].forward_transform(np.concatenate(y))
                y_norm_shuff = np.array(sorted(y_norm, key=lambda k: random.random()))
                Yt_hat = self.get_predict_samples(self.val_loader)
                ll = _log_lik(y_norm, Yt_hat, self.config["tau"]).mean()
                ll_shuff = _log_lik(y_norm_shuff, Yt_hat, self.config["tau"]).mean()
                scores["eval_ll"] = ll
                scores["eval_ll_shuff"] = ll_shuff
                print(self.get_mean_and_variance(self.val_loader))

            all_scores.append(scores)
        return all_scores

    def _get_sample(self, loader):
        y_hat = []
        for bidx, data in enumerate(loader):
            data = data.to(self.device)
            logit = self.model(data, do_dropout=True, drop_p=self.config["drop_p"])
            y_hat.append(logit.detach().cpu().numpy())
        y_hat = np.concatenate(y_hat,0)
        return y_hat

    def get_predict_samples(self, data_loader):
        Yt_hat = []
        for i in range(self.config["T"]):
            Yt_hat.append(self._get_sample(data_loader))
        return np.asarray(Yt_hat)

    def get_mean_and_variance(self, data_loader):
        # \mean{t in T} (\tau^-1 + y_hat_t^2) - \mean_{t in T}(y_hat_t)^2
        Yt_hat = self.get_predict_samples(data_loader)
        sigma2 = 1./self.config["tau"]
        var = (sigma2 + Yt_hat**2).mean(0) - Yt_hat.mean(0)**2
        return Yt_hat.mean(0), var




config = merge_dicts(DEFAULT_CONFIG, config)

if __name__ == "__main__":
    ray.init(memory=config["memory"])
    regressor_config = config["regressor_config"]
    # this will run train the model in a plain way
    # analysis = tune.run(**regressor_config)

    # this will fit the model directly
    rg = MCDropRegressor(regressor_config)
    print(rg.fit(rg.train_loader, rg.val_loader))

