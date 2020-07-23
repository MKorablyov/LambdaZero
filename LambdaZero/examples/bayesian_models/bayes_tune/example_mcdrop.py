import socket, os,sys, time, random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader

import ray
from ray import tune
from ray.rllib.utils import merge_dicts

from LambdaZero.utils import get_external_dirs, BasicRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.mpnn import config

from scipy.special import logsumexp


transform = LambdaZero.utils.Complete()
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


    # reg = config['length_scale'] ** 2 * (1 - dropout) / (2. * N * config['tau'])





    for bidx,data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        logits = model(data, do_dropout=True)
        targets_norm = config["normalizer"].forward_transform(targets)
        loss = F.mse_loss(logits, targets_norm)
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
        logits = model(data, do_dropout=True)
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
            #"target_norm": [-43.042, 7.057],
            "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
                                           #"brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
            "b_size": 50,
            "num_mc_samples": 10,

            "dataset": LambdaZero.inputs.BrutalDock,
            "dataset_config": {
                "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                "props": ["gridscore"],
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



def _log_lik(y, Yt_hat, tau=1.0):
    "computes log likelihood"
    ll = logsumexp(-0.5 * tau * (y[None] - Yt_hat) ** 2., 0)
    ll -= np.log(Yt_hat.shape[0])
    ll -= 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau)
    return ll







class MCDropRegressor(BasicRegressor):
    def __init__(self, regressor_config):
        super(BasicRegressor, self).__init__(regressor_config["config"])

    def fit(self, train_loader=None, val_loader=None):
        if train_loader is not None: self.train_loader = train_loader
        if val_loader is not None: self.val_loader = val_loader

        # todo allow ray native stopping
        scores = []

        for i in range(75):
            scores.append(self._train())
            print(scores[-1]["eval_mae"], scores[-1]["train_mae"])

            if i % 5 == 1:
                y = [getattr(s, self.config["target"]).cpu().numpy() for s in self.val_set]
                y_norm = self.config["normalizer"].forward_transform(np.concatenate(y))
                Yt_hat = self.get_predict_samples(self.val_loader)

                y_norm_shuff = np.array(sorted(y_norm, key=lambda k: random.random()))

                ll = _log_lik(y_norm, Yt_hat).mean()
                ll_shuff = _log_lik(y_norm_shuff, Yt_hat).mean()

                print("ll", ll, "ll_shuf", ll_shuff)


        return scores

    def _get_sample(self, loader):
        epoch_logits = []
        for bidx, data in enumerate(loader):
            data = data.to(self.device)
            logits = self.model(data, do_dropout=True)
            # fixme - keep dropout same for the dataset
            epoch_logits.append(logits.detach().cpu().numpy())
        epoch_logits = np.concatenate(epoch_logits, 0)
        return epoch_logits

    def get_predict_samples(self, data_loader):
        epoch_logits = []
        for i in range(self.config["num_mc_samples"]):
            epoch_logits.append(self._get_sample(data_loader))
        return np.asarray(epoch_logits)

    def get_mean_and_variance(self, data_loader):
        epoch_logits = self.get_predict_samples(data_loader)
        raise NotImplementedError("todo")

        #mean_hat = epoch_logits.mean(axis=0)
        #var_hat = epoch_logits.var(axis=0)

        # \mean{t in T} (\tau^-1 + y_hat_t^2) - \mean_{t in T}(y_hat_t)^2
        #return mean_hat, var_hat




config = merge_dicts(DEFAULT_CONFIG, config)

if __name__ == "__main__":
    ray.init(memory=config["memory"])
    regressor_config = config["regressor_config"]
    # this will run train the model in a plain way
    #analysis = tune.run(**regressor_config)


    rg = MCDropRegressor(regressor_config)
    print(rg.fit())
    #dataloader = DataLoader(regressor_config["config"]["dataset"](**regressor_config["config"]["dataset_config"])[:100])
    #mean, var = rg.get_mean_and_variance(dataloader)
    #print(mean,var)


    #train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)
    #train_set = DataLoader(Subset(dataset, train_idxs.tolist()), shuffle=True, batch_size=config["b_size"])

