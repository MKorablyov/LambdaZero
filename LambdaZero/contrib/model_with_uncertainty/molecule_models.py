import sys, time
from math import isclose
import ray
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
#from LambdaZero.models.torch_graph_models import MPNNet_Parametric, fast_from_data_list
from LambdaZero.inputs.inputs_op import _brutal_dock_proc
from torch_geometric.data import Batch
from LambdaZero.models import MPNNetDrop
from LambdaZero.inputs import random_split
from LambdaZero.contrib.inputs import ListGraphDataset
from .model_with_uncertainty import ModelWithUncertainty
from copy import deepcopy

def train_epoch(loader, model, optimizer, device):
    model.train()
    epoch_y = []
    epoch_y_hat = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        y_hat = model(data, do_dropout=True)
        loss = F.mse_loss(y_hat[:,0], data.y)
        loss.backward()
        optimizer.step()
        epoch_y.append(data.y.detach().cpu().numpy())
        epoch_y_hat.append(y_hat[:,0].detach().cpu().numpy())
    epoch_y = np.concatenate(epoch_y,0)
    epoch_y_hat = np.concatenate(epoch_y_hat, 0)
    return {"model/train_mse_loss":((epoch_y_hat-epoch_y)**2).mean()}


def val_epoch(loader, model, device):
    model.eval()
    epoch_y = []
    epoch_y_hat = []
    for bidx, data in enumerate(loader):
        data = data.to(device)
        y_hat = model(data, do_dropout=False)
        epoch_y.append(data.y.detach().cpu().numpy())
        epoch_y_hat.append(y_hat[:,0].detach().cpu().numpy())
    epoch_y = np.concatenate(epoch_y,0)
    epoch_y_hat = np.concatenate(epoch_y_hat, 0)
    return {"model/val_mse_loss":((epoch_y_hat-epoch_y)**2).mean()}


class MolMCDropGNN(ModelWithUncertainty):
    def __init__(self, train_epochs, batch_size, mpnn_config, lr, transform, num_mc_samples, device, logger):
        ModelWithUncertainty.__init__(self, logger)
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.mpnn_config = mpnn_config
        self.lr = lr
        self.transform = transform
        self.num_mc_samples = num_mc_samples

        self.device = device

    def fit(self,x,y):
        # initialize new model and optimizer
        model = MPNNetDrop(**self.mpnn_config)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # from many possible properties take molecule graph
        graphs = [m["mol_graph"] for m in x]
        graphs = deepcopy(graphs) # todo: I am forced to deepcopy graphs to prevent transform modyfying original graphs
        if self.transform is not None:
            graphs = [self.transform(g) for g in graphs]

        [setattr(graphs[i],"y", torch.tensor([y[i]])) for i in range(len(graphs))]
        train_idx, val_idx = random_split(len(graphs), [0.95, 0.05])
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        # do train epochs
        train_set = ListGraphDataset(train_graphs)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=True)
        val_set = ListGraphDataset(val_graphs)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=False)

        for i in range(self.train_epochs):
            train_metrics = train_epoch(train_loader, model, optimizer, self.device)
            val_metrics = val_epoch(val_loader, model, self.device)
            # todo: we want to see some convergence metrics, and logging every epoch is too much for wandb
        self.logger.log.remote(train_metrics)
        self.logger.log.remote(val_metrics)
        # update internal copy of the model
        #[delattr(graphs[i], "y") for i in range(len(graphs))]
        model.eval()
        self.model = model

    def update(self, x, y, x_new, y_new):
        mean, var = self.get_mean_and_variance(x_new)
        self.logger.log.remote({"model/acquired_mse_before_update":((np.array(y_new) - np.array(mean))**2).mean()})
        self.fit(x+x_new, y+y_new)
        mean, var = self.get_mean_and_variance(x_new)
        self.logger.log.remote({"model/acquired_mse_after_update": ((np.array(y_new) - np.array(mean)) ** 2).mean()})
        return None

    def get_mean_and_variance(self,x):
        y_hat_mc = self.get_samples(x, num_samples=self.num_mc_samples)
        return y_hat_mc.mean(1), y_hat_mc.var(1)

    def get_samples(self, x, num_samples):
        graphs = [m["mol_graph"] for m in x]
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,collate_fn=Batch.from_data_list)

        y_hat_mc = []
        for i in range(num_samples):
            y_hat_epoch = []
            for batch in dataloader:
                batch.to(self.device)
                y_hat_batch = self.model(batch, do_dropout=True)[:,0]
                y_hat_epoch.append(y_hat_batch.detach().cpu().numpy())
            y_hat_mc.append(np.concatenate(y_hat_epoch,0))
        y_hat_mc = np.stack(y_hat_mc,1)
        return y_hat_mc