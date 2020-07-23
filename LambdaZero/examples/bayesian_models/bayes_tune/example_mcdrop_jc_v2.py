import torch.nn as nn
import socket, os,sys, time
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops
from torch.utils.data.sampler import RandomSampler
import torch_geometric.transforms as T

import ray
from ray import tune
from ray.rllib.utils import merge_dicts

from tune_utils_v2 import BasicRegressor_v2
from LambdaZero.utils import get_external_dirs, BasicRegressor
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models
from LambdaZero.examples.mpnn import config
from scipy.special import logsumexp

from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set

transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "mpnn000"
config = getattr(config, config_name)

class MPNNetDrop(nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def __init__(self, num_feat=14, dim=64):
        super(MPNNetDrop, self).__init__()
        self.drop_prob = 0.5
        self.lin0 = nn.Linear(num_feat, dim)

        h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, h, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)
        
    def forward(self, data, do_dropout):
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
        out = nn.functional.dropout(out, p = self.drop_prob, training=do_dropout)
        return out.view(-1)

def train_epoch(loader, model, optimizer, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    dropout = model.drop_prob
    N = len(loader)

    T = 10000
    reg_term = config['length_scale']**2 * (1 - dropout) / (2. * N * config['tau'])
    reg_term_bias = config['length_scale'] ** 2 / (2. * N * config['tau'])
    
    metrics = {"loss":0, "mse": 0, "mae":0, "rmse": 0, "ll": 0, "tau_prime": 0}
    for bidx,data in enumerate(loader):
        data = data.to(device)
        #data = data[:3000]
        target = getattr(data, config["target"])
        optimizer.zero_grad()
        logit = model(data, do_dropout=True)

        MC_pred = np.mean(logit.detach().cpu().numpy())

        reg = 0
        reg_bias = 0
        for idx, param in enumerate(model.parameters()):
            if idx % 2 == 0:
                reg += reg_term * param.norm(2)
            else:
                reg_bias += reg_term_bias * param.norm(2)
        
        loss = F.mse_loss(logit, normalizer.normalize(target))
        loss.backward()
        optimizer.step()
        
        tau_prime = 0.5 * config['length_scale']/(2 * N) #set lambda = 1
        ll = (logsumexp(-0.5 * tau_prime * (target.detach().cpu().numpy() - logit.detach().cpu().numpy())**2., 0) - \
np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau_prime))  

        metrics["loss"] += loss.item() * data.num_graphs
        pred = normalizer.unnormalize(logit)
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

def get_predict_mean_and_variance_func():
    def prediction(x_test):
        predictions = []

        mean_x = np.mean(np.array(predictions), axis = 1)
        var_x = np.var(np.array(predictions), axis = 1)

        return var_x, mean_x

    return prediction

def rest_eval( rest_loader, regressor):
    regressor.eval()
    y_all = []
    out_all = []
    for i, (x, y) in enumerate(rest_loader):
        output = regressor(x)
        y_all.append(y.detach().cpu().numpy())
        out_all.append(1)

    y_all = np.concatenate(y_all, axis = 0)
    return y_all, out_all

def eval_epoch(loader, model, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.eval()
    metrics = {"loss":0,}
    epoch_targets = []
    epoch_preds = []

    for bidx,data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        logits = model(data, False)
        loss = F.mse_loss(logits, normalizer.normalize(targets))

        # log stuff                                                                                                                                                                  
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.unnormalize(logits).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets,0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])

    return metrics

aq_functions = ['random', 'greedy', 'egreedy', 'ucb', 'thompson']

DEFAULT_CONFIG = {
    "trainer": BasicRegressor_v2,
    "trainer_config": {
        "aq_function": aq_functions[0],
        "aq_batch_size": 32,
        "target": "gridscore",
        "target_norm": [-43.042, 7.057],
       # "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),#randsplit_Zinc15_260k.npy"),
        "dataset_split_path": osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy"),
        "b_size": 64,

        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
            "props": ["gridscore"],
            "transform": transform,
            #"file_names": ["Zinc15_2k"]#["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
            "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
        },
        #"target_norm": [-26.3, 12.3],
        #"dataset_split_path": osp.join(datasets_dir, "brutal_dock/d4/raw/randsplit_dock_blocks105_walk40_clust.npy"),
        #"b_size": 64,

        "tau": 1.0,
        "length_scale": 0.7,

        "model": MPNNetDrop,
        "model_config": {},

        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },

        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        'rest_eval': rest_eval
    },

    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,

    "stop": {"training_iteration": 200},
    "resources_per_trial": {
        "cpu": 4,  # fixme requesting all CPUs blocks additional call to ray from LambdaZero.input
        "gpu": 1.0
    },
    "num_samples":1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 250000000,
}


config = merge_dicts(DEFAULT_CONFIG, config)
name = 'lambda:_{}'.format(str(config['trainer_config']['length_scale']))

if __name__ == "__main__":
    ray.init(memory=config["memory"])

    analysis = tune.run(config["trainer"],
                        name = name,
                        config=config["trainer_config"],
                        stop=config["stop"],
                        resources_per_trial=config["resources_per_trial"],
                        num_samples=config["num_samples"],
                        checkpoint_at_end=config["checkpoint_at_end"],
                        local_dir=summaries_dir,
                        checkpoint_freq=config["checkpoint_freq"])

