import sys, time
from math import isclose
import ray
import torch
import torch.nn.functional as F
import numpy as np
import scipy
from torch.utils.data import DataLoader
#from LambdaZero.models.torch_graph_models import MPNNet_Parametric, fast_from_data_list
from LambdaZero.inputs.inputs_op import _brutal_dock_proc
from torch_geometric.data import Batch
from LambdaZero.models import MPNNetDrop
from LambdaZero.inputs import random_split
from LambdaZero.contrib.inputs import ListGraphDataset
from LambdaZero.contrib.model_with_uncertainty import ModelWithUncertainty
from LambdaZero.model_with_uncertainty import MolMCDropGNN
from LambdaZero.model_with_uncertainty.molecule_models import train_epoch, val_epoch
from LambdaZero.utils.utils_op import pearson_correlation
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


class MolMCDropGNNGP(MolMCDropGNN):
    def __init__(self, train_epochs, batch_size, num_mc_samples, device, logger):
        MolMCDropGNN.__init__(self, train_epochs, batch_size, num_mc_samples, device, logger)
        self.num_data_points_gp = 1000

    def fit(self, x, y):
        # initialize new model and optimizer
        self.model = MPNNetDrop(True, False, True, 0.1, 14)
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # from many possible properties take molecule graph
        graphs = [m["mol_graph"] for m in x]

        [setattr(graphs[i], "y", torch.tensor([y[i]])) for i in range(len(graphs))]  # this will modify graphs
        train_idx, val_idx = random_split(len(graphs), [0.95, 0.05])
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        # do train epochs
        train_set = ListGraphDataset(train_graphs)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=True)
        val_set = ListGraphDataset(val_graphs)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=Batch.from_data_list, shuffle=False)

        for i in range(self.train_epochs):
            metrics = train_epoch(train_loader, self.model, optimizer, self.device)
            self.logger.log.remote(metrics)
            metrics = val_epoch(val_loader, self.model, self.device)
            self.logger.log.remote(metrics)

        # fit a gp on the embeddings
        train_x = [x[i] for i in train_idx]
        embed = self.get_embed(train_x)[-self.num_data_points_gp:, :]
        self.gp = SingleTaskGP(embed, torch.tensor(y).unsqueeze(-1)[-self.num_data_points_gp:, :])
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(self.mll)

        # calculate correlation between true error and the var of gp using validation dataset
        epoch_pear_gp = []
        epoch_spar_gp = []
        for bidx, data in enumerate(val_loader):
            data = data.to(self.device)
            true_error = F.mse_loss(self.model(data, do_dropout=False)[:, 0], data.y, reduction='none').detach()
            gp_var = torch.tensor(self.gp(self.model.get_embed(data, do_dropout=False).detach().cpu()).stddev ** 2).to(
                    self.device)
            corr_gp = pearson_correlation(gp_var, true_error)
            epoch_pear_gp.append(corr_gp.detach().cpu().numpy())
            corr_gp = scipy.stats.spearmanr(gp_var.detach().cpu().numpy(), true_error.cpu().numpy())
            epoch_spar_gp.append(corr_gp)

        # update internal copy of the model
        [delattr(graphs[i], "y") for i in range(len(graphs))]
        self.model.eval()
        self.logger.log.remote({"pearson_corr_gp_var_true_error": np.array(epoch_pear_gp).mean(),
                                "spearman_corr_gp_var_true_error": np.array(epoch_spar_gp).mean()})

    def get_mean_and_variance(self,x):
        embed = self.get_embed(x)
        mean_x, var_x = self.gp(embed).mean.detach().numpy(), self.gp(embed).stddev.detach().numpy() ** 2
        self.logger.log.remote({"model/var_gp": var_x})
        return mean_x, var_x
        

