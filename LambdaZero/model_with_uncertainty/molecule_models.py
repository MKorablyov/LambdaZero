import sys, time

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
# from LambdaZero.models.torch_graph_models import MPNNet_Parametric, fast_from_data_list
from torch_geometric.data import Batch
from ray.tune.integration.wandb import wandb_mixin
import wandb
from LambdaZero.models import MPNNetDrop
from LambdaZero.contrib.inputs import ListGraphDataset
from LambdaZero.contrib.model_with_uncertainty import ModelWithUncertainty


def train_epoch(loader, model, optimizer, device):
    model.train()
    epoch_y = []
    epoch_y_hat = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        y_hat = model(data, do_dropout=True)
        loss = F.mse_loss(y_hat[:, 0], data.y)
        loss.backward()
        optimizer.step()
        epoch_y.append(data.y.detach().cpu().numpy())
        epoch_y_hat.append(y_hat[:, 0].detach().cpu().numpy())
    epoch_y = np.concatenate(epoch_y, 0)
    epoch_y_hat = np.concatenate(epoch_y_hat, 0)
    # todo: make more detailed metrics including examples being acquired
    return {"train_mse_loss": ((epoch_y_hat - epoch_y) ** 2).mean()}


@wandb_mixin
class MolMCDropGNN(ModelWithUncertainty):
    def __init__(self, train_epochs, batch_size, num_mc_samples, device, logger):
        ModelWithUncertainty.__init__(self, logger)
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.num_mc_samples = num_mc_samples
        self.device = device

    def fit(self, x, y):
        # initialize new model and optimizer
        self.model = MPNNetDrop(True, False, True, 0.1, 16)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # from many possible properties take molecule graph
        graphs = [m["mol_graph"] for m in x]
        [setattr(graphs[i], "y", torch.tensor([y[i]])) for i in range(len(graphs))]

        # do train epochs
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=True)

        for i in range(self.train_epochs):
            metrics = train_epoch(dataloader, self.model, self.optimizer, self.device)
            # todo: add weight decay etc.
            self.logger.log.remote(metrics)

            print("train GNNDrop", metrics)
            # wandb.log(metrics)

    def get_mean_and_variance(self, x):
        y_hat_mc = self.get_samples(x, num_samples=self.num_mc_samples)
        return y_hat_mc.mean(1), y_hat_mc.var(1)

    def get_samples(self, x, num_samples):
        graphs = [m["mol_graph"] for m in x]
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list)

        y_hat_mc = []
        for i in range(num_samples):
            y_hat_epoch = []
            for batch in dataloader:
                batch.to(self.device)
                y_hat_batch = self.model(batch, do_dropout=True)[:, 0]
                y_hat_epoch.append(y_hat_batch.detach().cpu().numpy())
            y_hat_mc.append(np.concatenate(y_hat_epoch, 0))
        y_hat_mc = np.stack(y_hat_mc, 1)
        return y_hat_mc

    def posterior(self, x, observation_noise=False):
        mean_m, variance_m = self.get_mean_and_variance(x)
        if observation_noise:
            pass

        mvn = MultivariateNormal(mean_m.squeeze(), torch.diag(variance_m.squeeze() + 1e-6))
        return GPyTorchPosterior(mvn)

    def get_embed(self, x):
        graphs = [m["mol_graph"] for m in x]
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list)
        # y_hat_epoch = []
        embed = None
        for batch in dataloader:
            batch.to(self.device)
            y_hat_batch = self.model.get_embed(batch, do_dropout=True)
            # y_hat_epoch.append(y_hat_batch.detach().cpu().numpy())
            if embed is None:
                embed = y_hat_batch.detach().cpu()
            else:
                embed = torch.cat((embed, y_hat_batch.detach().cpu()), dim=0)

        return embed

