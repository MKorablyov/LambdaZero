import socket, os,sys, time, random
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

# import bayes_vs import bayes_models
from bayes_vs import chem_ops
from torch_geometric.data import DataLoader

from scipy.special import logsumexp

import ray
from ray import tune
from ray.rllib.utils import merge_dicts

from LambdaZero.utils import get_external_dirs, BasicRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.mpnn import config
import scipy.misc

from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set

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
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    dropout = model.drop_prob
    N = len(loader)
    #import pdb; pdb.set_trace()
    T = config['T']
    reg_term = config['length_scale']**2 * (1 - dropout) / (2. * N * config['tau'])
    reg_term_bias = config['length_scale'] ** 2 / (2. * N * config['tau'])
    
    metrics = {"loss":0, "mse": 0, "mae":0, "rmse": 0, "ll": 0, "tau_prime": 0}
    for bidx,data in enumerate(loader):
        data = data.to(device)
        #data = data[:3000]
        target = getattr(data, config["target"])
        optimizer.zero_grad()
        means = []
        varis = []
        for _ in range(config['N']):
            logit = model(data, do_dropout=config['drop_layers'], mlp_drop=config['drop_mlp'])

            means.append(np.mean(logit.detach().cpu().numpy()))
            varis.append(np.var(logit.detach().cpu().numpy()))
        
        MC_pred = np.mean(means)
        MC_var = np.mean(varis)

        reg = 0
        reg_bias = 0
        for idx, param in enumerate(model.parameters()):
            if idx % 2 == 0:
                reg += reg_term * param.norm(2)
            else:
                reg_bias += reg_term_bias * param.norm(2)
        
        loss = F.mse_loss(logit, target) + reg + reg_bias
        loss.backward()
        optimizer.step()
        
        tau_prime = 0.5 * config['length_scale']/(2 * N) #set lambda = 1
        ll = (logsumexp(-0.5 * tau_prime * (target.detach().cpu().numpy() - logit.detach().cpu().numpy())**2., 0) - \
np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau_prime))  

        metrics["loss"] += loss.item() * data.num_graphs
        pred = logit
        metrics["ll"] += ll
        metrics["mse"] += ((target - pred) ** 2).sum().item()
        metrics["mae"] += ((target - pred).abs()).sum().item()

    #metrics["rmse"] = metrics["rmse"] / len(loader.dataset)
    metrics["ll"] = metrics["ll"] / len(loader.dataset)
    #metrics["rmse"] = metrics["rmse"] / len(loader.dataset)
    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["rmse"] = metrics["mse"]**0.5
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    return metrics


def eval_epoch(loader, model, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.eval()
    metrics = {"loss":0, "mse": 0, "mae":0, "rmse": 0, "ll": 0, "tau_prime": 0}
    epoch_targets = []
    epoch_preds = []

    dropout = model.drop_prob
    N = len(loader)

    T = config['T']
    reg_term = config['length_scale']**2 * (1 - dropout) / (2. * N * config['tau'])
    reg_term_bias = config['length_scale'] ** 2 / (2. * N * config['tau'])

    for bidx,data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        
        # loss = F.mse_loss(logits, targets)

        means = []
        varis = []
        losses = []
        # for _ in range(config['N']):
        logit = model(data, False, False)
        loss = F.mse_loss(logit, targets)
        MC_pred = np.mean(logit.detach().cpu().numpy())
        MC_var = np.var(logit.detach().cpu().numpy())
        
        # MC_pred = np.mean(means)
        # MC_var = np.mean(varis)
        # loss = np.mean(losses)

        reg = 0
        reg_bias = 0
        for idx, param in enumerate(model.parameters()):
            if idx % 2 == 0:
                reg += reg_term * param.norm(2)
            else:
                reg_bias += reg_term_bias * param.norm(2)

        # log stuff                                                                                                                                                                  
        # metrics["loss"] += loss.item() * data.num_graphs

        tau_prime = 0.5 * config['length_scale']/(2 * N) #set lambda = 1
        ll = (logsumexp(-0.5 * tau_prime * (targets.detach().cpu().numpy() - logit.detach().cpu().numpy())**2., 0) - \
np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau_prime))  

        metrics["loss"] += loss.item() * data.num_graphs
        pred = logit
        metrics["ll"] += ll
        metrics["mse"] += ((targets - pred) ** 2).sum().item()
        metrics["mae"] += ((targets - pred).abs()).sum().item()


        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(logit.detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets,0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()
    metrics["rmse"] = metrics["mse"]**0.5
    metrics["ll"] = metrics["ll"] / epoch_targets.shape[0]

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])

    return metrics

def rest_evalu(rest_loader, regressor):
    regressor.eval()
    y_all = []
    out_all = []
    for i, (x, y) in enumerate(rest_loader):
        output = regressor(x)
        y_all.append(y.detach().cpu().numpy())
        out_all.append(1)

    y_all = np.concatenate(y_all, axis = 0)
    return y_all, out_all




def _compute_ll(samples, vals):
    "computes log likelihood"
    sqerr = (samples - vals)**2
    sqerr = torch.Tensor(sqerr)
    ll = torch.logsumexp(-.5 * sqerr,0).numpy().mean()   
    return ll

# todo: dropout (p= 0.01, 0.03, 0.09, 0.5)
# todo: lengthscale (0.01, 0.03, ??? )
# T: 
# dropout_hyperparameter (drop_layers=True, drop_mlp=True)

# todo: add BLL (from John's code)

class MPNNetDrop(nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def __init__(self, dropout, num_feat=14, dim=64):
        super(MPNNetDrop, self).__init__()
        self.drop_prob = dropout
        self.lin0 = nn.Linear(num_feat, dim)

        h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, h, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)
        
    def forward(self, data, do_dropout, mlp_drop):
        out = nn.functional.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = nn.functional.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = nn.functional.relu(self.lin1(out))
        out = nn.functional.dropout(out, p = self.drop_prob, training=do_dropout)
        out = self.lin2(out)
        out = nn.functional.dropout(out, p = self.drop_prob, training=mlp_drop)
        return out.view(-1)

    def fit(self, train_loader=None, val_loader=None):
        if train_loader is not None: self.train_loader = train_loader
        if val_loader is not None: self.val_loader = val_loader

        # todo allow ray native stopping
        scores = []

        for i in range(75):
            scores.append(self._train())
            print(scores[-1]["eval_mae"], scores[-1]["train_mae"])

            if i % 5 == 1:
                val_targets = [getattr(s, self.config["target"]).cpu().numpy() for s in self.val_set]
                val_targets_norm = self.config["normalizer"].forward_transform(np.concatenate(val_targets))
                ll =  _compute_ll(self.get_predict_samples(self.val_loader, val_targets_norm[None,:]))
                print("LL", ll)

                shuffled_targets = np.array(sorted(val_targets_norm, key=lambda k: random.random()))
                shuffled_ll = _compute_ll(self.get_predict_samples(self.val_loader), shuffled_targets[None,:])
                print("shuffled LL", shuffled_ll)

                #print(preds.shape)

                #mu_hat, sigma_hat = self.get_mean_and_variance(self.val_loader)

                #logp = LambdaZero.utils.logP(mu_hat, sigma_hat, val_targets_norm)

                # shuffled_targets = sorted(val_targets_norm, key=lambda k: random.random())
                # shuffled_logp = LambdaZero.utils.logP(mu_hat, sigma_hat, shuffled_targets)
                #
                # print("MAE", np.abs(mu_hat - val_targets_norm).mean(),
                #       "shuffled MAE", np.abs(mu_hat - shuffled_targets).mean(),
                #       "logp", logp.mean(), "shuffle logP", shuffled_logp.mean())

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
        mean_hat = epoch_logits.mean(axis=0)
        var_hat = epoch_logits.var(axis=0)
        return mean_hat, var_hat

aq_functions = ['random', 'greedy', 'egreedy', 'ucb', 'thompson']

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "N": 50,
        "T": 3,
        "aq_function": aq_functions[0],
        "aq_batch_size": 32,
        "target": "gridscore",
        "target_norm": [-43.042, 7.057],
       # "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),#randsplit_Zinc15_260k.npy"),
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
        "b_size": 32,
        "num_mc_samples": 10,

        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore"],
            "transform": transform,
           # "file_names": ["Zinc15_2k"]#["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
            "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
        },
        #"target_norm": [-26.3, 12.3],
        #"dataset_split_path": osp.join(datasets_dir, "brutal_dock/d4/raw/randsplit_dock_blocks105_walk40_clust.npy"),
        #"b_size": 64,

        "tau": 1.0,
        "drop_layers" : True,
        "drop_mlp" : True,
        "model": MPNNetDrop,
        "model_config": {},

        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch
        #'rest_eval': rest_eval
    },

    "num_samples":1,
    "dropout_val": [0.1, 0.3, 0.5, 0.8],
    "log_scale_val": [0.01, 0.03, 0.09, 0.27, 0.81, 2.43],
    
    "checkpoint_at_end": False,
    "checkpoint_freq": 250000000,

    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,

    "stop": {"training_iteration": 200},
    "resources_per_trial": {
        "cpu": 4,  # fixme requesting all CPUs blocks additional call to ray from LambdaZero.input
        "gpu": 1.0
    }
}

config = merge_dicts(DEFAULT_CONFIG, config)
dropout_val = config["dropout_val"]
log_scale_val = config["log_scale_val"]

if __name__ == "__main__":
    ray.init(memory=config["memory"])

    for d in dropout_val:
        for l in log_scale_val:
            name = 'dropout_val:_{} log_scale_val:_{}'.format(str(d), str(l))
            config['trainer_config']['model_config']['dropout'] = d
            config['trainer_config']['length_scale'] = l

            analysis = tune.run(config["trainer"],
                                name = name,
                                config=config["trainer_config"],
                                stop=config["stop"],
                                resources_per_trial=config["resources_per_trial"],
                                num_samples=config["num_samples"],
                                checkpoint_at_end=config["checkpoint_at_end"],
                                local_dir=summaries_dir,
                                checkpoint_freq=config["checkpoint_freq"])


    # regressor_config = config["regressor_config"]
    # # this will run train the model in a plain way
    # #analysis = tune.run(**regressor_config)


    # rg = MCDropRegressor(regressor_config)
    # print(rg.fit())
    #dataloader = 
DataLoader(regressor_config["config"]["dataset"](**regressor_config["config"]["dataset_config"])[:100])
    #mean, var = rg.get_mean_and_variance(dataloader)
    #print(mean,var)


    #train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)
    #train_set = DataLoader(Subset(dataset, train_idxs.tolist()), shuffle=True, batch_size=config["b_size"])
