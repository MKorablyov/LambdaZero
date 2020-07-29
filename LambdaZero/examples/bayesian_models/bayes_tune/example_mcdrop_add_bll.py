import socket, os,sys, time, random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import torch.nn as nn
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
from matplotlib import pyplot as plt
from sklearn import linear_model

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

    N = len(loader)
    lambd = config['lambda'] #config['lengthscale'] ** 2 * (1 - config["drop_p"]) / (2. * N * config['tau'])


    for bidx,data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        logits = model(data, do_dropout=True, drop_p=config["drop_p"] )
        targets_norm = config["normalizer"].forward_transform(targets)

        reg_weight = 0
        reg_bias = 0

        for idx, param in enumerate(model.parameters()):
            if idx % 2 == 0:
                reg_weight += lambd * param.norm(2)
            else:
                reg_bias += (lambd/(1-config["drop_p"])) * param.norm(2)
        loss = F.mse_loss(logits, targets_norm) + reg_weight + reg_bias
        # reg_loss = lambd * torch.stack([(p ** 2).sum() for p in model.parameters()]).sum()
        # loss = F.mse_loss(logits, targets_norm) + reg_loss
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
            "lambda": 1e-8, #, 1e-6, 1e-4, 1e-2, 1],#5.0,
            "T": 10, #[10,100,1000, 10000], #10,
            "drop_p": 0.1, #[0.1, 0.3, 0.5, 0.7, 0.9],
            "lengthscale": 1e-2, #[1e-1, 1.0, 10, 100],


            "dataset": LambdaZero.inputs.BrutalDock,
            "dataset_config": {
                "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                "props": ["gridscore"],
                "transform": transform,
                "file_names": ["Zinc15_2k"]
                #"file_names": ["Zinc15_2k"], #["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
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


class MCDropRegressor(BasicRegressor):
    def __init__(self, regressor_config):
        super(BasicRegressor, self).__init__(regressor_config["config"])

    def fit(self, train_loader=None, val_loader=None):
        if train_loader is not None: self.train_loader = train_loader
        if val_loader is not None: self.val_loader = val_loader

        self.N = len(self.train_loader.dataset)
        

        # todo allow ray native stopping


        all_scores = []
        for i in range(50):
            scores = self._train()
            if i % 25 == 0:
                y = [getattr(s, self.config["target"]).cpu().numpy() for s in self.val_set]
                y_norm = self.config["normalizer"].forward_transform(np.concatenate(y))
                y_norm_shuff = np.array(sorted(y_norm, key=lambda k: random.random()))
                Yt_hat = self.get_predict_samples(self.val_loader, self.model)
                ll = _log_lik(y_norm, Yt_hat, self.config, self.N).mean()
                ll_shuff = _log_lik(y_norm_shuff, Yt_hat, self.config, self.N).mean()
                scores["eval_ll"] = ll
                scores["eval_ll_shuff"] = ll_shuff


        print('mcdrop ll: {}'.format(ll))
        print('mcdrop shuff ll: {}'.format(ll_shuff))

        #New model to drop the last layer, use the second to last layer as bayesian embedding

        newmodel = MPNNetDrop2()
        newmodel.to(self.device)
        newmodel.load_state_dict(self.model.state_dict())

        del newmodel.lin2

        #get all embeddings
        emb_train = []
        emb_train_tar = []
        for i, j in enumerate(self.train_loader):
            emb_train.append(newmodel(j.to(self.device),do_dropout=True, drop_p=self.config["drop_p"]).detach().cpu().numpy())
            targets = getattr(j, self.config["target"])
            targets_norm = self.config["normalizer"].forward_transform(targets)
            emb_train_tar.append(targets_norm.detach().cpu().numpy())


        emb_val = []
        emb_val_tar = []
        for i, j in enumerate(self.val_loader):
            emb_val.append(newmodel(j.to(self.device),do_dropout=True, drop_p=self.config["drop_p"]).detach().cpu().numpy())
            targets = getattr(j, self.config["target"])
            targets_norm = self.config["normalizer"].forward_transform(targets)
            emb_val_tar.append(targets_norm.detach().cpu().numpy())

        emb_train = np.concatenate(emb_train, 0)
        emb_val = np.concatenate(emb_val, 0)
        emb_train_tar = np.concatenate(emb_train_tar, 0)
        emb_val_tar = np.concatenate(emb_val_tar, 0)

        #Train a bayesian ridge regressor on the embeddings
        clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)
        clf.fit(emb_train, emb_train_tar) 

        pred = clf.predict(emb_val)
        scores = _compute_metrics(emb_val_tar, pred, self.config["normalizer"])

        #Get ll from bayesian model
        y = [getattr(s, self.config["target"]).cpu().numpy() for s in self.val_set]
        y_norm = self.config["normalizer"].forward_transform(np.concatenate(y))
        y_norm_shuff = np.array(sorted(y_norm, key=lambda k: random.random()))
        Yt_hat = pred 
        ll = _log_lik(y_norm, Yt_hat, self.config, self.N).mean()
        ll_shuff = _log_lik(y_norm_shuff, Yt_hat, self.config, self.N).mean()
        scores["eval_ll"] = ll
        scores["eval_ll_shuff"] = ll_shuff

        print(ll)
        print(ll_shuff)

        return scores

    def _get_sample(self, loader, model, drop=True):
        y_hat = []
        for bidx, data in enumerate(loader):
            data = data.to(self.device)
            logit = model(data, do_dropout=drop, drop_p=self.config["drop_p"])
            y_hat.append(logit.detach().cpu().numpy())
        y_hat = np.concatenate(y_hat,0)
        return y_hat

    def get_predict_samples(self, data_loader, model, drop=True):
        Yt_hat = []
        for i in range(self.config["T"]):
            Yt_hat.append(self._get_sample(data_loader, model, drop))
        return np.asarray(Yt_hat)

    def get_mean_and_variance(self, data_loader):

        # Scalar version:
        Yt_hat = self.get_predict_samples(data_loader)
        tau = get_tau(self.config, self.N)
        sigma2 = 1./tau
        var = (sigma2 + Yt_hat**2).mean(0) - Yt_hat.mean(0)**2
        return Yt_hat.mean(0), var

#changes MPNNetDrop forward pass to delete the final linear layer
class MPNNetDrop2(LambdaZero.models.MPNNetDrop):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def forward(self, data, do_dropout, drop_p):
        out = nn.functional.dropout(nn.functional.relu(self.lin0(data.x)), training = do_dropout, p=drop_p)
        h = out.unsqueeze(0)

        for i in range(3):
            m = nn.functional.dropout(nn.functional.relu(self.conv(out, data.edge_index, data.edge_attr)), training = do_dropout, p=drop_p)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = nn.functional.dropout(nn.functional.relu(self.lin1(out)), training = do_dropout, p=drop_p)
        out = nn.functional.dropout(out, training=do_dropout, p=drop_p)

        return out

config = merge_dicts(DEFAULT_CONFIG, config)

if __name__ == "__main__":
    ray.init(memory=config["memory"])
    regressor_config = config["regressor_config"]
    # this will run train the model in a plain way
    # analysis = tune.run(**regressor_config)

    # this will fit the model directly    
    rg = MCDropRegressor(regressor_config)
    #print('experiment: lambda = {}\t T = {} \r\t\t drop_prob = {}\t length_scale = {}'.format(lambd, T, p, length_scale))
    print(rg.fit())

