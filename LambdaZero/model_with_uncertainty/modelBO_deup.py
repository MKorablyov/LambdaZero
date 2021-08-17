import sys, time
from math import isclose
import ray
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import explained_variance_score
from torch_geometric.data import Batch
from LambdaZero.models import MPNNetDrop
from LambdaZero.inputs import random_split
from LambdaZero.contrib.data import ListGraphDataset
from LambdaZero.contrib.modelBO import ModelBO, train_epoch, val_epoch, MolMCDropGNN
from .molecule_models import get_mcdrop_var_batch
from sklearn.model_selection import KFold
from copy import deepcopy
from .metrics import create_uncertainty_metrics, uncertainty_metrics


def train_deup_epoch(loader, e_model, e_optimizer, feature_to_deup, device):
    e_model.train()
    epoch_e, epoch_e_hat = [], []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        e_optimizer.zero_grad()
        for b in [0, 1]:
            if feature_to_deup is 'xb':
                e_hat = e_model(data, do_dropout=True, out_sample=b)
            else:
                deup_input = []
                if 'v' in feature_to_deup:
                    if b == 0: deup_input += [data.in_sample_mcdrop_var.cpu().numpy()]
                    elif b == 1: deup_input += [data.out_sample_mcdrop_var.cpu().numpy()]
                if 'd' in feature_to_deup: deup_input += [data.density.detach().cpu().numpy()]
                if 'b' in feature_to_deup: deup_input += [np.array([b]*len(data.y))]
                deup_input = torch.transpose(torch.tensor(deup_input), 0, 1)

                e_hat = e_model(deup_input.to(device))
            if b == 0:
                e_loss = F.mse_loss(e_hat[:, 0], data.in_sample_e)
                e_loss.backward()
            elif b == 1:
                e_loss = F.mse_loss(e_hat[:, 0], data.out_sample_e)
                e_loss.backward()

            e_optimizer.step()

        epoch_e.append(data.out_sample_e.cpu().numpy())
        epoch_e_hat.append(e_hat[:, 0].detach().cpu().numpy())

    # todo: make more detailed metrics including examples being acquired
    return {"model/train_deup_mse_loss": ((np.concatenate(epoch_e_hat, 0) - np.concatenate(epoch_e, 0)) ** 2).mean(),}


def val_deup_epoch(loader, model, e_model, feature_to_deup, num_mc_samples, device, i):
    model.eval()
    e_model.eval()

    epoch_metrics_deup = create_uncertainty_metrics()
    epoch_metrics_mc = create_uncertainty_metrics()
    for bidx, data in enumerate(loader):
        data = data.to(device)
        y_hat = model(data, do_dropout=False)
        _, mcdrop_var = get_mcdrop_var_batch(model, data, num_mc_samples)
        if feature_to_deup is 'xb':
            e_hat = e_model(data, do_dropout=False, out_sample=1)
        else:
            deup_input = []
            if 'v' in feature_to_deup: deup_input += [mcdrop_var]
            if 'd' in feature_to_deup: deup_input += [data.density.detach().cpu().numpy()]
            if 'b' in feature_to_deup: deup_input += [np.array([1]*len(data.y))]
            deup_input = torch.transpose(torch.tensor(deup_input), 0, 1)

            e_hat = e_model(deup_input.to(device))

        mcdrop_uncertainty_metrics = uncertainty_metrics(y_hat[:, 0].detach().cpu().numpy(), data.y.detach().cpu().numpy(), mcdrop_var)
        deup_uncertainty_metrics = uncertainty_metrics(y_hat[:, 0].detach().cpu().numpy(), data.y.detach().cpu().numpy(),  e_hat[:, 0].detach().cpu().numpy())

        for key in deup_uncertainty_metrics.keys():
            epoch_metrics_deup[key].append(deup_uncertainty_metrics[key])
        for key in mcdrop_uncertainty_metrics.keys():
            epoch_metrics_mc[key].append(mcdrop_uncertainty_metrics[key])


    return {"model/val_mse_loss":np.array(epoch_metrics_deup['mse']).mean(),
            "model/val_deup_mse_loss": np.array(epoch_metrics_deup['mse_deup']).mean(),
            "model/val_correlation_deup": np.array(epoch_metrics_deup['pear_corr']).mean(),
            "model/val_correlation_mcdrop": np.array(epoch_metrics_mc['pear_corr']).mean(),
            "model/val_spearman_corr_deup": np.array(epoch_metrics_deup['spearman_corr']).mean(),
            "model/val_spearman_corr_mcdrop": np.array(epoch_metrics_mc['spearman_corr']).mean(),
            "model/val_ll_deup": np.array(epoch_metrics_deup['neg_log_lik']).mean(),
            "model/val_ll_mcdrop": np.array(epoch_metrics_mc['neg_log_lik']).mean(),
            }


def create_e_net(mpnn_config, feature_to_deup, lr, device):
    # initialize new model and optimizer
    model = MPNNetDrop(**mpnn_config).to(device)
    if feature_to_deup is 'xb':
        e_model = MPNNetDropDeup(**mpnn_config).to(device)
    else:
        e_model = nn.Sequential(
            nn.Linear(in_features=len(feature_to_deup), out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=1),
            nn.Softplus(),
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    e_optimizer = torch.optim.Adam(e_model.parameters(), lr=lr/10)

    return model, optimizer, e_model, e_optimizer


class MolMCDropGNNDeup(MolMCDropGNN):
    def __init__(self, train_epochs, batch_size, mpnn_config, lr, transform, num_mc_samples, log_epoch_metrics, device,
                 logger):
        MolMCDropGNN.__init__(self, train_epochs, batch_size, mpnn_config, lr, transform, num_mc_samples, log_epoch_metrics, device,
                 logger)

        self.feature_to_deup = 'xb'
        self.load_densities = True
        self.fine_tune = True


        if self.fine_tune:
            self.model, self.optimizer, self.e_model, self.e_optimizer = create_e_net(
                self.mpnn_config, self.feature_to_deup, self.lr, self.device)

    def fit(self,x,y):

        # initialize new model and optimizer
        if not self.fine_tune:
            self.model, self.optimizer, self.e_model, self.e_optimizer = create_e_net(
                self.mpnn_config, self.feature_to_deup, self.lr, self.device)

        if 'd' in self.feature_to_deup:
            densities, valid_masks = self.get_lstm_densities(x, self.load_densities)
            # drop invalid inputs
            x = [i for i, m in zip(x, valid_masks) if m == True]
            y = [i for i, m in zip(y, valid_masks) if m == True]
            [setattr(x[i]["mol_graph"], "density", torch.tensor([densities[i]])) for i in
             range(len(x))]  # this will modify graphs
            self.valid_masks = valid_masks
        # from many possible properties take molecule graph
        graphs = self._preprocess(x)

        [setattr(graphs[i], "y", torch.tensor([y[i]])) for i in range(len(graphs))]

        train_idx, val_idx = random_split(len(graphs), [0.9, 0.1])
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        val_set = ListGraphDataset(val_graphs)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=False)

        train_loader, iid_metrics, val_metrics = self.build_dataset(train_graphs, val_loader)
        # Train DEUP
        print('Training the main predictor...')
        for i in range(self.train_epochs):
            train_deup_metrics = train_deup_epoch(train_loader, self.e_model, self.e_optimizer,
                                                  self.feature_to_deup, self.device)
            val_deup_metrics = val_deup_epoch(val_loader, self.model, self.e_model, self.feature_to_deup,
                                              self.num_mc_samples, self.device, i)
            if self.log_epoch_metrics:
                self.logger.log.remote(train_deup_metrics)
                self.logger.log.remote(val_deup_metrics)

        self.logger.log.remote(iid_metrics)
        self.logger.log.remote(train_deup_metrics)
        self.logger.log.remote(val_metrics)
        self.logger.log.remote(val_deup_metrics)

    def get_mean_and_variance(self,x):
        y_hat_mc = self.get_samples(x, num_samples=self.num_mc_samples)
        return y_hat_mc.mean(1), y_hat_mc.var(1)

    def get_samples(self, x, num_samples):
        graphs = self._preprocess(x)
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

    def get_deup_error_samples(self, x):
        if 'd' in self.feature_to_deup:
            densities, valid_masks = self.get_lstm_densities(x, self.load_densities)
            # drop invalid inputs
            x = [i for i, m in zip(x, valid_masks) if m == True]
            [setattr(x[i]["mol_graph"], "density", torch.tensor([densities[i]])) for i in range(len(x))]
        else:
            valid_masks = [True] * len(x)
        self.valid_masks = valid_masks
        graphs = [m["mol_graph"] for m in x]
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list)
        e_hat_epoch = []
        for data in dataloader:
            data.to(self.device)
            # todo: find a better way to compute mcdropout var for a batch
            mcdrop_var = get_mcdrop_var_batch(self.model, data, self.num_mc_samples)[1]
            if self.feature_to_deup == 'xb':
                e_hat = self.e_model(data, do_dropout=False, out_sample=1)
            else:
                deup_input = []
                if 'v' in self.feature_to_deup:
                    deup_input += [mcdrop_var]
                if 'd' in self.feature_to_deup:
                    deup_input += [data.density.detach().cpu().numpy()]
                if 'b' in self.feature_to_deup:
                    deup_input += [np.array([1] * (data.batch.max().item()+1))]
                deup_input = torch.transpose(torch.tensor(deup_input), 0, 1)

                e_hat = self.e_model(deup_input.to(self.device))
            e_hat_epoch.append(e_hat.detach().cpu().numpy().ravel())
        return np.concatenate(e_hat_epoch, 0)

    def get_lstm_densities(self, x, load_densities):
        if not load_densities:
            smiles = [m["smiles"] for m in x]
            densities, valid_masks = [], []
            smiles_loader = DataLoader(smiles, batch_size=self.batch_size, collate_fn=None, shuffle=False)
            for bidx, data in enumerate(smiles_loader):
                if bidx % 1000 == 0:
                    print(f'density estimation, idx:{bidx}')
                d, val_mask = probability_for_batch(data, self.device)
                densities += d.detach().cpu().numpy().tolist()
                valid_masks += val_mask
                np.save(f'{summaries_dir}densities_split_Zinc20_docked_neg_randperm_30k.npy',
                    np.array(densities))
            np.save(f'{summaries_dir}valid_masks_split_Zinc20_docked_neg_randperm_30k.npy',
                    np.array(valid_masks))
        else:
            densities = np.load(f'{summaries_dir}densities_split_Zinc20_docked_neg_randperm_30k.npy').tolist()
            valid_masks = np.load(
                f'{summaries_dir}valid_masks_split_Zinc20_docked_neg_randperm_30k.npy').tolist()

        return densities, valid_masks

    def build_dataset(self, train_graphs, val_loader):
        n_splits = 2
        kf = KFold(n_splits=n_splits)
        models = [None] * n_splits
        optimizers = [None] * n_splits
        fold = 0
        for iid_index, ood_index in kf.split(train_graphs):
            print(f'Training the error predictor on fold number {fold+1} ...')
            model = MPNNetDrop(**self.mpnn_config).to(self.device)
            models[fold] = model
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            optimizers[fold] = optimizer
            iid_graph = [train_graphs[i] for i in iid_index]
            ood_graph = [train_graphs[i] for i in ood_index]
            iid_set = ListGraphDataset(iid_graph)
            iid_loader = DataLoader(iid_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=False)
            ood_set = ListGraphDataset(ood_graph)
            ood_loader = DataLoader(ood_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=False)

            for i in range(self.train_epochs):
                iid_metrics = train_epoch(iid_loader, model, optimizer, self.device)
                val_metrics = val_epoch(val_loader, model, self.device)
                if self.log_epoch_metrics:
                    self.logger.log.remote(iid_metrics)
                    self.logger.log.remote(val_metrics)

            model.eval()

            in_sample_error = []
            mcdrop_vars = []
            for bidx, data in enumerate(iid_loader):
                data = data.to(self.device)
                y_hat = model(data, do_dropout=False)
                in_sample_error += F.mse_loss(y_hat[:, 0], data.y, reduction='none').detach().cpu().numpy().tolist()
                mcdrop_vars += get_mcdrop_var_batch(model, data, self.num_mc_samples)[1].tolist()
            [setattr(train_graphs[iid_index[i]], "in_sample_e", torch.tensor([in_sample_error[i]])) for i in
             range(len(iid_graph))]
            [setattr(train_graphs[iid_index[i]], "in_sample_mcdrop_var", torch.tensor([mcdrop_vars[i]])) for i in
             range(len(iid_graph))]

            out_sample_error = []
            mcdrop_vars = []
            for bidx, data in enumerate(ood_loader):
                data = data.to(self.device)
                y_hat = model(data, do_dropout=False)
                out_sample_error += F.mse_loss(y_hat[:, 0], data.y, reduction='none').detach().cpu().numpy().tolist()
                mcdrop_vars += get_mcdrop_var_batch(model, data, self.num_mc_samples)[1].tolist()
            [setattr(train_graphs[ood_index[i]], "out_sample_e", torch.tensor([out_sample_error[i]])) for i in
             range(len(ood_graph))]
            [setattr(train_graphs[ood_index[i]], "out_sample_mcdrop_var", torch.tensor([mcdrop_vars[i]])) for i in
             range(len(ood_graph))]

            fold += 1
        self.model = deepcopy(model)
        final_ood_set = ListGraphDataset(train_graphs)
        final_ood_loader = DataLoader(final_ood_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list,
                                      shuffle=True)

        return final_ood_loader, iid_metrics, val_metrics


class MPNNetDropDeup(MPNNetDrop):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    Adapted for error predictor: Statioanry features are concatenated to embeddings
    """
    def __init__(self, drop_last, drop_data, drop_weights, drop_prob, num_feat=14, dim=64, out_dim=1):
        super().__init__(drop_last, drop_data, drop_weights, drop_prob, num_feat, dim, out_dim)

        self.lin2 = nn.Sequential(nn.Linear(dim+1, out_dim),
                                  nn.Softplus())

    def forward(self, data, do_dropout, out_sample=0):
        embed = self.get_embed(data, do_dropout)
        embed = torch.cat((embed, torch.tensor([out_sample] * embed.size(0)).to(embed.device).unsqueeze(-1)), dim=1)
        out = self.lin2(embed)
        return out

config_MolMCDropGNNDeup_v1 = {
    "train_epochs":75,
    "batch_size":75,
    "mpnn_config":{
        "drop_last":True,
        "drop_data":False,
        "drop_weights":True,
        "drop_prob":0.1,
        "num_feat":14
    },
    "lr":1e-3,
    "transform":None,
    "num_mc_samples":10,
    "log_epoch_metrics":False,
    "device":"cuda"
}