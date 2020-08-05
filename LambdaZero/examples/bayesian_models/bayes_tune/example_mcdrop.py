import socket, os,sys, time, random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric import transforms as T
from torch_geometric.data import DataLoader

import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.rllib.models.catalog import ModelCatalog



from LambdaZero.utils import get_external_dirs, BasicRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
#from LambdaZero.examples.mpnn import config

from scipy.special import logsumexp
from sklearn import linear_model
from matplotlib import pyplot as plt

datasets_dir, programs_dir, summaries_dir = get_external_dirs()
transform = T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()])


def _epoch_metrics(epoch_targets_norm, epoch_logits, normalizer):
    epoch_targets = normalizer.itfm(epoch_targets_norm)
    epoch_preds = normalizer.itfm(epoch_logits)
    metrics = {}
    #metrics["loss"] = metrics["loss"] / epoch_targets.shape[0] todo
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()
    # ranking
    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])
    return metrics

def get_tau(config, N):
    tau = (1 - config["drop_p"]) * (config["lengthscale"]**2) / (2 * N * config["lambda"])
    return tau

def _log_lik(y, Yt_hat, config, N):
    "computes log likelihood"
    # ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat) ** 2., 0)
    # - np.log(T)
    # - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau))
    tau = get_tau(config, N)
    ll = logsumexp(-0.5 * tau * (y[None] - Yt_hat) ** 2., 0)
    ll -= np.log(Yt_hat.shape[0])
    ll -= 0.5 * np.log(2 * np.pi)
    ll += 0.5 * np.log(tau)
    return ll


def train_epoch(loader, model, optimizer, device, config):
    model.train()
    epoch_targets_norm = []
    epoch_logits = []

    for bidx,data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        logits = model(data, do_dropout=True, drop_p=config["drop_p"])
        targets_norm = config["normalizer"].tfm(targets)
        reg_loss = config['lambda'] * torch.stack([(p ** 2).sum() for p in model.parameters()]).sum()
        loss = F.mse_loss(logits, targets_norm) + reg_loss
        loss.backward()
        optimizer.step()

        epoch_targets_norm.append(targets_norm.detach().cpu().numpy())
        epoch_logits.append(logits.detach().cpu().numpy())

    epoch_targets_norm = np.concatenate(epoch_targets_norm,0)
    epoch_logits = np.concatenate(epoch_logits, 0)
    scores = _epoch_metrics(epoch_targets_norm, epoch_logits, config["normalizer"])
    return scores


def sample_logits(loader, model, device, config, num_samples, do_dropout):
    sample_logits = []
    for i in range(num_samples):
        epoch_logits = []
        for bidx, data in enumerate(loader):
            data = data.to(device)
            logit = model(data, do_dropout=do_dropout, drop_p=config["drop_p"])
            epoch_logits.append(logit.detach().cpu().numpy())
        sample_logits.append(np.concatenate(epoch_logits, 0))
    return np.stack(sample_logits,0)


def sample_targets(loader, config):
    epoch_targets = [getattr(d, config["target"]).cpu().numpy() for d in loader.dataset]
    norm_targets = config["normalizer"].tfm(np.concatenate(epoch_targets,0))
    return norm_targets


def eval_epoch(loader, model, device, config):
    logits = sample_logits(loader, model, device, config, 1, False)[0]
    norm_targets = sample_targets(loader, config)
    scores = _epoch_metrics(norm_targets, logits, config["normalizer"])
    return scores

def eval_uncertainty(loader, model, device, config, N):
    norm_targets = sample_targets(loader, config)
    logits = sample_logits(loader, model, device, config, num_samples=config["T"], do_dropout=True)

    ll = _log_lik(norm_targets, logits, config, N).mean()
    shuff_targets = np.array(sorted(norm_targets, key=lambda k: random.random()))
    shuf_ll = _log_lik(shuff_targets, logits, config, N).mean()
    return {"ll":ll, "shuff_ll":shuf_ll}


def _dataset_creator(config):
    # make dataset
    dataset = config["dataset"](**config["dataset_config"])
    train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)

    train_set = Subset(dataset, train_idxs.tolist())
    val_set = Subset(dataset, val_idxs.tolist())
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config["b_size"])
    val_loader = DataLoader(val_set, batch_size=config["b_size"])
    return train_loader, val_loader


class MCDrop(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config["dataset_creator"] is not None:
            self.train_loader, self.val_loader = config["dataset_creator"](config)
        self.N = len(self.train_loader.dataset)

        # make model
        self.model = config["model"](**config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](self.model.parameters(), **config["optimizer_config"])
        # make epochs
        self.train_epoch, self.eval_epoch = config["train_epoch"], config["eval_epoch"]
        # todo: add embeddings

    def _train(self):
        train_scores = self.train_epoch(self.train_loader, self.model, self.optim, self.device, self.config)
        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = self.eval_epoch(self.val_loader, self.model, self.device, self.config)
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)

        if self._iteration % 15 == 1:
            print("iteration", self._iteration)
            eval_scores = eval_uncertainty(self.val_loader, self.model, self.device, self.config, self.N)
            eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
            scores = dict(list(scores.items()) + eval_scores)

        return scores

    def fit(self, train_loader, val_loader):
        # update internal dataset
        self.train_loader, self.val_loader = train_loader, val_loader
        # make a new model
        self.model = self.config["model"](**self.config["model_config"]) # todo: reset?
        self.model.to(self.device)

        # todo allow ray native stopping
        all_scores = []
        for i in range(2):
            scores = self._train()
            all_scores.append(scores)
        return all_scores

    def get_samples(self, loader, num_samples):
        return sample_logits(loader, self.model, self.device, self.config, num_samples)

    def get_mean_and_variance(self, loader):
        # \mean{t in T} (\tau^-1 + y_hat_t^2) - \mean_{t in T}(y_hat_t)^2
        Yt_hat = self.get_samples(loader, self.config["T"])
        tau = get_tau(self.config, self.N)
        sigma2 = 1./tau
        var = (sigma2 + Yt_hat**2).mean(0) - Yt_hat.mean(0)**2
        return Yt_hat.mean(0), var








# todo: dropout (p= 0.01, 0.03, 0.09, 0.5)
# todo: lengthscale (0.01, 0.03, ??? )
# dropout_hyperparameter (drop_layers=True, drop_mlp=True)
# todo: add BLL (from John's code)
# todo: I want to be able to do proper tune logging
# todo: I don't want to assume I have a dataset when initializing MCDrop
# todo: I want to have a proper fit() function; maybe that could run tune in the background




DEFAULT_CONFIG = {
    "regressor_config":{
        "run_or_experiment": MCDrop,
        "config": {
            "target": "gridscore",
            "dataset_creator": _dataset_creator,
            "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
                                           #"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
            "dataset": LambdaZero.inputs.BrutalDock,
            "dataset_config": {
                "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                "props": ["gridscore", "smi"],
                "transform": transform,
                "file_names": ["Zinc15_2k"],  # ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],

            },
            "b_size": 40,
            "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057]),

            "lambda": 1e-6,
            "T": 10,
            "drop_p": 0.1,
            "lengthscale": 1e-2,

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



#if len(sys.argv) >= 2: config_name = sys.argv[1]
#else: config_name = "mpnn000"
#config = getattr(config, config_name)
#config = merge_dicts(DEFAULT_CONFIG, config)
config = DEFAULT_CONFIG




if __name__ == "__main__":
    ray.init(memory=config["memory"])
    # this will run train the model in a plain way

    tune.run(**config["regressor_config"])



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

        clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)
        clf.fit(train_fps, train_targets_norm)

        train_logits = clf.predict(train_fps)
        val_logits, val_std  = clf.predict(val_fps, return_std=True)

        # def _epoch_metrics(epoch_targets_norm, epoch_logits, normalizer):
        train_scores = _epoch_metrics(train_targets_norm, train_logits, config["normalizer"])
        val_scores = _epoch_metrics(val_targets_norm, val_logits, config["normalizer"])
        ll = -0.5 * np.mean(np.log(2 * np.pi * val_std ** 2) + ((val_targets_norm - val_logits) ** 2 / val_std ** 2))

        print("train:", train_scores)
        print("eval:", val_scores, "LL:", ll)


    bll_on_fps(config["regressor_config"]["config"])


