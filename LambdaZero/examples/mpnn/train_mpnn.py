import socket, os, time
import numpy as np
import os.path as osp
import math
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
#from torch_geometric.data import DataLoader
from torch.utils.data import Sampler
import time
import sys

from torch_geometric.data import Data, Batch
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T
import ray
from ray import tune

from LambdaZero.examples.mpnn import config
from LambdaZero.utils import get_external_dirs
import LambdaZero.inputs
import LambdaZero.utils
import LambdaZero.models

from dime_net import DimeNet

from ray.rllib.utils import merge_dicts

from custom_dataloader import DL

import matplotlib.pyplot as plt


def train_epoch(loader, model, optimizer, device, config, max_val):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.train()

    metrics = {"loss":0, "mse": 0, "mae":0}

    i = 0
    for bidx,data in enumerate(loader):
        i += 1
        # compute y_hat and y
        data = data.to(device)

        optimizer.zero_grad()
        
        if config['model'] == "dime":
            logit = model(data.x, data.pos, data.edge_index)
        else:
            logit = model(data)

        pred = (logit * target_norm[1]) + target_norm[0]
        y = getattr(data, target)
        
        if config['loss'] == "L2":
            loss = F.mse_loss(logit, (y - target_norm[0]) / target_norm[1])
        elif config['loss'] == "L1":
            loss = F.l1_loss(logit, (y - target_norm[0]) / target_norm[1])
        else:
            raise ValueError("Not Implemented type loss: " + config['loss'])

        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item() * data.num_graphs
        metrics["mse"] += ((y - pred) ** 2).sum().item()
        metrics["mae"] += ((y - pred).abs()).sum().item()
    
    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)

    return metrics


def eval_epoch(loader, model, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0, "mae_regret": 0, "mse_regret": 0, "median_regret": 0}
    
    running_preds_true_energy = th.tensor([],requires_grad=False)
    running_preds = th.tensor([],requires_grad=False)
    running_gc = th.tensor([],requires_grad=False)

    for bidx, data in enumerate(loader):

        # compute y_hat and y
        data = data.to(device)
        logit = model(data)

        pred = (logit * target_norm[1]) + target_norm[0]
        y = getattr(data, target)
        
        cpu_y = y.cpu()
        cpu_pred = pred.cpu()
    
        highest_ys = th.topk(y.cpu(), config['num_rank'])
        running_gc = th.cat((running_gc, highest_ys[0]), 0)
        running_gc = th.topk(running_gc, config['num_rank'])[0]

        highest_preds = th.topk(cpu_pred, config['num_rank'])

        running_preds = th.cat((running_preds, highest_preds[0]), 0)
        running_preds_true_energy = th.cat((running_preds_true_energy, cpu_y[highest_preds[1]]), 0)

        running_preds = th.topk(running_preds, config['num_rank'])

        running_preds_true_energy = running_preds_true_energy[running_preds[1]].detach()
        running_preds = running_preds[0].detach()
        
        loss = F.mse_loss(logit, (y - target_norm[0]) / target_norm[1])
        metrics["loss"] += loss.item() * data.num_graphs
        metrics["mse"] += ((y - pred) ** 2).sum().item()
        metrics["mae"] += ((y - pred).abs()).sum().item()    

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    metrics["median_regret"] = abs(np.median(running_preds_true_energy) - np.median(running_gc)).item()
    metrics["mae_regret"] = F.l1_loss(running_preds_true_energy, running_gc).item()
    metrics["mse_regret"] = F.mse_loss(running_preds_true_energy, running_gc).item()

    return metrics


class BasicRegressor(tune.Trainable):
    def _setup(self, config):

        self.config = config
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # make model
        if not config['model'] == "dime":
            self.model = LambdaZero.models.MPNNet()
        else:
            self.model = DimeNet(in_channels=14, hidden_channels=128,
                                out_channels=1, num_blocks=6, num_bilinear=8, num_spherical=7,
                                num_radial=6, cutoff=5)

        print("Hit here")

        self.model.to(self.device)
        self.optim = th.optim.Adam(self.model.parameters(), lr=config["lr"])

        # load dataset
        if not config['model'] == "dime":
            dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                                    props=config["molprops"],
                                                    transform=config["transform"],
                                                    file_names=config["file_names"])
        else:
            dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                                    props=config["molprops"],
                                                    transform=config["transform"],
                                                    file_names=config["file_names"],
                                                    proc_func=LambdaZero.inputs.tpnn_proc)

        
        # split dataset
        split_path = osp.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
        train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)

        bsize = config["b_size"]
        sampler = None
        
        self.max_val = 0

        train_idxs = train_idxs[:-1]
        
        if config['use_sampler']:
            train_dataset = dataset[th.tensor(train_idxs)]

            energies = th.tensor([-graph.gridscore.item() for graph in train_dataset], dtype=th.float64)
            energies = energies + energies.min() + 1

            energies = energies.pow(config['pow'])
            energies_prob = energies/energies.sum()

            sampler = th.utils.data.sampler.WeightedRandomSampler(energies_prob, len(train_dataset))

        train_subset = Subset(dataset, train_idxs.tolist())
        val_subset = Subset(dataset, val_idxs.tolist())
        test_subset = Subset(dataset, test_idxs.tolist())
        
        if sampler == None:
            self.train_set = DL(train_subset, batch_size=bsize, shuffle=True)
        else:
            self.train_set = DL(train_subset, batch_size=bsize, samp=sampler)

        self.val_set = DL(val_subset, batch_size=bsize)
        self.test_set = DL(test_subset, batch_size=bsize)

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]


    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config, self.max_val)
        eval_scores = self.eval_epoch(self.val_set, self.model,  self.device, self.config)
        # rename to make scope
        train_scores = [("train_" + k, v) for k,v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        th.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(th.load(checkpoint_path))


if len(sys.argv) == 2: 
    config_name = sys.argv[1]
    config = getattr(config, config_name)
else:
    config_name = "mpnn_" + sys.argv[1] + "_" + sys.argv[2]
    config = {
            "trainer_config":{
                "loss": sys.argv[2],
                "pow": float(sys.argv[1]),
                "use_sampler": True
            }
        }

transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()


DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "dataset_root": os.path.join(datasets_dir, "brutal_dock/sars-cov-2"),
        "targets": ["gridscore"],
        "target_norms": [[-26.3, 12.3]],
        "file_names": ["Zinc15_260k_0","Zinc15_260k_1","Zinc15_260k_2","Zinc15_260k_3"],
        "transform": transform,
        "split_name": "randsplit_Zinc15_260k",
        "lr": 0.001,
        "b_size": 64,
        "dim": 64,
        "num_epochs": 5,
        "num_rank": 15,
        "molprops": ["gridscore"],
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "model": "mpnn"
        },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 250000000,
    "stop": {"training_iteration": 100},
}

config = merge_dicts(DEFAULT_CONFIG, config)

if __name__ == "__main__":

    if True:
        ray.init()

        analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop={"training_iteration":100}, #EarlyStop(),
                        resources_per_trial={
                           "cpu": 12,
                           "gpu": 1.0
                        },
                        num_samples=1,
                        checkpoint_at_end=True,
                        local_dir=config['summaries_dir']+"/final/",
                        loggers=None,
                        name=config_name,
                        checkpoint_freq=100000)

        raise ValueError("Done with the trial.")
