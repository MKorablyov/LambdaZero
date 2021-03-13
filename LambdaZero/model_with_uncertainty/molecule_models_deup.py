import sys, time

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
# from LambdaZero.models.torch_graph_models import MPNNet_Parametric, fast_from_data_list
from torch_geometric.data import Batch
from ray.tune.integration.wandb import wandb_mixin
import wandb
from LambdaZero.models import MPNNetDrop
from LambdaZero.contrib.inputs import ListGraphDataset
from LambdaZero.contrib.model_with_uncertainty import ModelWithUncertainty


def train_epoch(loader, model, optimizer, e_model, e_optimizer, num_mc_samples, device):
    model.train()
    e_model.train()
    epoch_y = []
    epoch_y_hat = []
    epoch_e = []
    epoch_e_hat = []
    epoch_corr_deup = []
    epoch_corr_mc = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()

        y_hat = model(data, do_dropout=False)
        loss = F.mse_loss(y_hat[:, 0], data.y)
        loss.backward()
        optimizer.step()

        e_optimizer.zero_grad()
        y_hat_mc = []
        for i in range(num_mc_samples):
            y_hat_epoch = []
            y_hat_batch = model(data, do_dropout=True)[:, 0]
            y_hat_epoch.append(y_hat_batch.detach().cpu().numpy())
            y_hat_mc.append(np.concatenate(y_hat_epoch, 0))
        y_hat_mc = np.stack(y_hat_mc, 1)
        e_hat = e_model(torch.tensor(y_hat_mc.var(1)).to(device).unsqueeze(-1))

        e_loss = F.mse_loss(e_hat[:, 0], F.mse_loss(y_hat[:, 0], data.y, reduction='none').detach())
        e_loss.backward()
        e_optimizer.step()

        # Calculate correlation between true error and predicted uncertainty
        error_true = F.mse_loss(y_hat[:, 0], data.y, reduction='none').detach()
        var_mc = torch.tensor(y_hat_mc.var(1)).to(device)
        error_deup = e_hat[:, 0]
        error_true = error_true - torch.mean(error_true)
        var_mc = var_mc - torch.mean(var_mc)
        error_deup = error_deup - torch.mean(error_deup)
        corr_mc = torch.sum(error_true * var_mc) / (
                torch.sqrt(torch.sum(error_true ** 2) + 1e-6) * torch.sqrt(torch.sum(var_mc ** 2) + 1e-6))
        corr_deup = torch.sum(error_true * error_deup) / (
                    torch.sqrt(torch.sum(error_true ** 2) + 1e-6) * torch.sqrt(torch.sum(error_deup ** 2) + 1e-6))

        epoch_y.append(data.y.detach().cpu().numpy())
        epoch_y_hat.append(y_hat[:, 0].detach().cpu().numpy())
        epoch_e.append(F.mse_loss(y_hat[:, 0], data.y, reduction='none').detach().cpu().numpy())
        epoch_e_hat.append(e_hat[:, 0].detach().cpu().numpy())
        epoch_corr_deup.append(corr_deup.detach().cpu().numpy())
        epoch_corr_mc.append(corr_mc.detach().cpu().numpy())

    epoch_y = np.concatenate(epoch_y, 0)
    epoch_y_hat = np.concatenate(epoch_y_hat, 0)
    epoch_e = np.concatenate(epoch_e, 0)
    epoch_e_hat = np.concatenate(epoch_e_hat, 0)
    epoch_corr_deup = np.array(epoch_corr_deup)
    epoch_corr_mc = np.array(epoch_corr_mc)
    # todo: make more detailed metrics including examples being acquired
    return {"train_mse_loss": ((epoch_y_hat - epoch_y) ** 2).mean(),
            "train_epistemic_mse_loss": ((epoch_e_hat - epoch_e) ** 2).mean(),
            "correlation_deup_true_error": epoch_corr_deup.mean(),
            "correlation_mc_var_true_error": epoch_corr_mc.mean()}


@wandb_mixin
class MolMCDropGNNDeup(ModelWithUncertainty):
    def __init__(self, train_epochs, batch_size, num_mc_samples, device, logger):
        ModelWithUncertainty.__init__(self, logger)
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.num_mc_samples = num_mc_samples
        self.device = device

    def fit(self, x, y):
        # initialize new model and optimizer
        self.model = MPNNetDrop(True, False, True, 0.1, 14).to(self.device)
        self.e_model = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
            nn.Softplus(),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.e_optimizer = torch.optim.Adam(self.e_model.parameters(), lr=1e-3)

        # from many possible properties take molecule graph
        graphs = [m["mol_graph"] for m in x]
        [setattr(graphs[i], "y", torch.tensor([y[i]])) for i in range(len(graphs))]

        # do train epochs
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=True)

        for i in range(self.train_epochs):
            metrics = train_epoch(dataloader, self.model, self.optimizer,
                                  self.e_model, self.e_optimizer, self.num_mc_samples, self.device)
            # todo: add weight decay etc.
            self.logger.log.remote(metrics)

            print("train GNNDrop", metrics)
            # wandb.log(metrics)

    def update(self, x, y, x_new, y_new):
        mean, var = self.get_mean_and_variance(x_new)
        self.logger.log.remote({"model/mse_before_update":((np.array(y_new) - np.array(mean))**2).mean()})
        self.fit(x+x_new, y+y_new)
        mean, var = self.get_mean_and_variance(x_new)
        self.logger.log.remote({"model/mse_after_update": ((np.array(y_new) - np.array(mean)) ** 2).mean()})
        return None

    def get_mean_and_variance(self, x):
        y_hat_mc = self.get_samples(x, num_samples=self.num_mc_samples)
        ep_error = self.get_deup_error_samples(x)
        self.logger.log.remote({"model/var_deup": ep_error, "model/var_mcdrop": y_hat_mc.var(1)})

        return y_hat_mc.mean(1), ep_error

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
        embed = None
        for batch in dataloader:
            batch.to(self.device)
            y_hat_batch = self.model.get_embed(batch, do_dropout=True)
            if embed is None:
                embed = y_hat_batch.detach().cpu()
            else:
                embed = torch.cat((embed, y_hat_batch.detach().cpu()), dim=0)

        return embed

    def get_deup_error_samples(self, x):
        graphs = [m["mol_graph"] for m in x]
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list)

        e_hat_epoch = []
        for batch in dataloader:
            batch.to(self.device)
            # todo: find a better way to compute mcdropout var for a batch
            y_hat_mc = []
            for i in range(self.num_mc_samples):
                y_hat_epoch = []
                y_hat_batch = self.model(batch, do_dropout=True)[:, 0]
                y_hat_epoch.append(y_hat_batch.detach().cpu().numpy())
                y_hat_mc.append(np.concatenate(y_hat_epoch, 0))
            y_hat_mc = np.stack(y_hat_mc, 1)

            e_hat_batch = self.e_model(torch.tensor(y_hat_mc.var(1)).to(self.device).unsqueeze(-1))[:, 0]
            e_hat_epoch.append(e_hat_batch.detach().cpu().numpy())

        return np.concatenate(e_hat_epoch, 0)

