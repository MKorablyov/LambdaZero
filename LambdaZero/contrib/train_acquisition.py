import numpy as np
from ray import tune

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from blitz.modules import BayesianLinear
#from blitz.utils import variational_estimator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import LambdaZero.chem


def evaluate_regression(regressor,
                        X,
                        y,
                        samples=100,
                        std_multiplier=2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

class UCB:
    # todo: acquisition function - maybe create classes
    #  AcqusitionFunction; ModelWithUncertainty
    def __init__(self):
        self.model = BayesianRegressor(1024, 1)
        self.seen_x, self.seen_y = None, None
        self.val_x, self.val_y = None, None

    def update_with_seen(self, x, y):
        # self.seen_x +=x
        # self.model_with_uncertainty.fit(x,y, self.val_x, self.val_y)
        pass

    def acqusition_values(self, x):
        # mean, var = self.model.get_mean_and_variance(molecules)
        # return mean + kappa * var
        pass

    def acquire_batch(self, x, discounts, aq_values=None):
        # if aq_values = self.compute_acquisition_values(x)
        # aq_values_ = aq_values[top_k]
        # return idx
        pass


class UCBTrainable(UCB, tune.Trainable):
    # load dataset of fingerprints

    # self.dataset = load_dataset()
    # _train()

        # return {"acc":0.1}
    pass


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

import csv
with open('/home/mkkr/1_step_docking_results_qed0.5.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        print(row[0].split(",")[1])
        #print(','.join(row))
























#regressor = BayesianRegressor(1024, 1)

# class UCBTrainer(UCB, tune.trainable):
    # _init():
    #   seen_x, seen_y, val_x, val_y, unseen_x, unseen_y = .....
    # def train()
    #   idx  = self.acquire_batch(unseen_x)
    #   x_, y_ = unseen[idx], unseen[idx]
    #   self.update_with_seen(x_, y_)

# tune run reference here:
# https://github.com/MKorablyov/LambdaZero/blob/master/LambdaZero/examples/bayesian_models/bayes_tune/UCB.py

