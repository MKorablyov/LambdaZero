# trying out gpytorch regression

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

import numpy as np
from ray import tune
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import LambdaZero.chem
from LambdaZero.chem import build_mol, mpnn_feat
from torch_sparse import coalesce
from torch_geometric.data import Data, InMemoryDataset

import csv
from torch.utils.data import TensorDataset, DataLoader
import itertools




train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size())

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

training_iter = 50
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

## for evaluating stuff:
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))


## for plotting stuff:
with torch.no_grad():
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    lower, upper = observed_pred.confidence_region()
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

def _mol_to_graph(atmfeat, coord, bond, bondfeat, props={}):
    """convert to PyTorch geometric module"""
    natm = atmfeat.shape[0]
    # transform to torch_geometric bond format; send edges both ways; sort bonds
    atmfeat = torch.tensor(atmfeat, dtype=torch.float32)
    edge_index = torch.tensor(np.concatenate([bond.T, np.flipud(bond.T)], axis=1), dtype=torch.int64)
    edge_attr = torch.tensor(np.concatenate([bondfeat, bondfeat], axis=0), dtype=torch.float32)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, natm, natm)
    # make torch data
    if coord is not None:
        coord = torch.tensor(coord, dtype=torch.float32)
        data = Data(x=atmfeat, pos=coord, edge_index=edge_index, edge_attr=edge_attr, **props)
    else:
        data = Data(x=atmfeat, edge_index=edge_index, edge_attr=edge_attr, **props)
    return data

def mol_to_graph(smiles, props={}, num_conf=1, noh=True, feat="mpnn"):
    """mol to graph convertor"""
    mol, _, _ = LambdaZero.chem.build_mol(smiles, num_conf=num_conf, noh=noh)
    if feat == "mpnn":
        atmfeat, coord, bond, bondfeat = mpnn_feat(mol)
    else:
        raise NotImplementedError(feat)
    graph = _mol_to_graph(atmfeat, coord, bond, bondfeat, props)
    return graph


smiles_list = []
scores_list = []
graphs_list = []
i = 0
with open('/home/mkkr/1_step_docking_results_qed0.5.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in itertools.islice(reader, 10):
        splitter = row[0].split(",")
        print(splitter[1])
        print(splitter[2])
        if splitter[1] != 'smiles':
            smiles_list.append(splitter[1])
            graphs_list.append(mol_to_graph(splitter[1]))
            scores_list.append(splitter[2])