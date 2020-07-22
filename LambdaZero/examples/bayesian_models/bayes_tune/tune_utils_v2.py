import numpy as np
import os, time, os.path as osp
import pandas as pd
import random
import torch as th
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
from blitz.modules import BayesianLinear
import ray
from ray import tune
# from torch.utils.tensorboard import SummaryWriter

class BasicRegressor_v2(tune.Trainable):
    def _setup(self, config):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load dataset
        dataset = config["dataset"](**config["dataset_config"])

        # split dataset
        train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)

        self.train_set = DataLoader(Subset(dataset, train_idxs.tolist()), shuffle=True, batch_size=config["b_size"])
        self.val_set = DataLoader(Subset(dataset, val_idxs.tolist()), batch_size=config["b_size"])
        self.test_set = DataLoader(Subset(dataset, test_idxs.tolist()), batch_size=config["b_size"])

        # make model
        self.model = config["model"](**config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](self.model.parameters(), **config["optimizer_config"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]
        self.rest_eval = config['rest_eval']

        # acquisition function class
        self.acquirer = Acquirer(config, device)
        self.regressor = config["model"](**config["model_config"])
        self.regressor.to(self.device)
        # self.writer = SummaryWriter('logs/')

###
       # self.config = config
       # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       # device = self.device



        # # load and spilt the data
        # dataset = FingerprintDataset(config, device)
        # self.train_set, _ = dataset.train_set()
        # _, self.test_loader = dataset.test_set()

        # self.top_mol_dict = {}
        # self.top_batch_dict = {}

        # self.regressor = ModelWithUncertainty(1024, 1)
        # self.regressor = self.regressor.to(self.device)

        # self.trainer = Trainer(self.config, torch.nn.MSELoss(), torch.optim.Adam(self.regressor.parameters(), lr=self.config["lr"]))

        self.is_aquired = np.zeros(len(self.train_set),dtype=np.bool)
        self.is_aquired[np.random.choice(np.arange(len(self.train_set)), self.config['aq_batch_size'])] = True

        # self.aq_fxns = {'random': self.acquirer.random_aq, 'greedy': self.acquirer.egreedy_aq, 'egreedy': self.acquirer.egreedy_aq, 'ucb': self.acquirer.ucb_aq, 'thompson': self.acquirer.thompson_aq}
        # self.aq_fxn = self.aq_fxns[config['aq_function']]
        # # make epochs
        # self.train_epoch = config["train_epoch"]
        # self.mse = []
        # self.mae = []

    def _train(self):
        name = self.config['aq_function']

        if name == 'egreedy':
            self.acquirer.noise_std = 5
            self.acquirer.noise_mean = 2.5
        else:
            self.acquirer.noise_std = 1
            self.acquirer.noise_mean = 0

        top_mol = []
        top_batch = []

        test_losses = []
        train_losses = []

        # while np.count_nonzero(is_aquired) < len(self.train_set) and np.count_nonzero(is_aquired) < self.config['sample_size']:
        aq_set = Subset(self.train_set, np.where(self.is_aquired)[0])
        aq_loader = torch.utils.data.DataLoader(aq_set, batch_size=self.config['aq_batch_size'])
        
        # train model
        # for i in range(self.train_epoch):
        trn_loss = self.train_epoch(self.regressor, aq_loader)

        tst_loss, tst_mae = self.eval_epoch(self.test_loader, self.regressor)
        test_losses.append(tst_loss)
        train_losses.append(trn_loss)
        self.mse.append(tst_loss)
        self.mae.append(tst_mae)
        # acquire more data
        rest_set = Subset(self.train_set, np.where(~self.is_aquired)[0])
        rest_loader = DataLoader(rest_set, batch_size=self.config['aq_batch_size'])

        ys, output = self.rest_eval(aq_loader, self.regressor) 
        output_indices = np.argsort(ys) 
        ys = ys[output_indices]
        top_pred = ys[0]
        top_mol.append(top_pred)

        # writer.add_scalar('Values {}'.format(name), top_pred)

        aq_indices = self.aq_fxn(rest_loader, self.regressor)
        self.is_aquired[aq_indices] = True
        #print(aq_indices)
        batch_set = Subset(self.train_set, aq_indices)      
        batch_ys = [y.detach().cpu().numpy() for (_, y) in batch_set]
        top_batch.append(np.amin(np.array(batch_ys)))
        # self.writer.add_scalar('top_mol', np.array(top_pred))
        # self.writer.add_scalar('top_batch', np.amin(np.array(batch_ys)))
        # self.writer.add_scalar('Test MSE', np.array(tst_loss))
        # self.writer.add_scalar('Test MAE', np.array(tst_mae))
        
        trn_loss = trn_loss.detach().cpu().numpy().item()
        #import pdb; pdb.set_trace()
        # self.writer.close()
        return {'train_top_mol' : top_pred, 'train_top_batch' : np.amin(np.array(batch_ys)),
                'Test MSE': tst_loss, 'Test MAE':tst_mae, 'Train MSE':trn_loss}


    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.regressor.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.regressor.load_state_dict(torch.load(checkpoint_path))

class Acquirer:
  def __init__(self, config, device, noise_std = 1, noise_mean = 0):
    self.batch_aq_num = config['aq_batch_size']
    self.device = device
    self.kappa = 0.8
    self.noise_std = noise_std
    self.noise_mean = noise_mean
    self.config = config

  def egreedy_aq(self,loader, model):
      if self.config['mcdrop']:
          model.train()
      else:
          model.eval()
      
      preds_all = []
      for bidx, (x, y) in enumerate(loader):
          x, y = x.to(self.device), y.to(self.device)
          output = model(x)
          preds_all.append(output.detach().cpu().numpy())
      preds = np.concatenate(preds_all, axis = 0)
      preds = torch.from_numpy(preds).float().to(self.device)
      dist = torch.distributions.normal.Normal(self.noise_mean, self.noise_std).sample(preds.shape).to(self.device)
      preds = preds + dist
      aq_order = np.argsort(np.concatenate(preds.detach().cpu().numpy(), axis=0))

      return aq_order[:self.batch_aq_num]
      
  
  def thompson_aq(self, loader, model):
      if self.config['mcdrop']:
          model.train()
      else:
          model.eval()
      eval_func = model.get_predict_sample()
      aq_batch = []

      all_scores = []
      all_batch = []
      all_indices = []
      j = 0
      for i, (x, y) in enumerate(loader):
        # ^ this is the loop which you parallelize with eg Ray
        all_batch.append(x)
        score_on_batch = eval_func(x)
        all_scores.append(score_on_batch[0])
        all_indices.append(range(j * self.batch_aq_num, (j+1) * self.batch_aq_num))
        j += 1
      
      next_point_to_query = np.argmax(all_scores)
      return all_indices[next_point_to_query]

  
  def ucb_aq(self, loader, model):
    mu_var = model.get_predict_mean_and_variance_func()
    all_scores = []
    #all_batch = []
    #all_indices = []
    #j = 0
    
    for i, (x, y) in enumerate(loader):
      # ^ this is the loop which you parallelize with eg Ray
      #all_batch.append(x)
      mean_on_batch, var_on_batch = mu_var(x)
      scores = np.add(mean_on_batch, self.kappa * var_on_batch)
      #score_on_batch = mean_on_batch + self.kappa * var_on_batch
      scores = np.concatenate(scores, axis = 0)
      all_scores.append(scores)
      #all_indices.append(range(j * self.batch_aq_num, (j+1) * self.batch_aq_num))
      #j += 1
    #import pdb; pdb.set_trace()
    all_scores = np.concatenate(all_scores, axis = 0)
    next_point_to_query = np.argsort(all_scores)
    return next_point_to_query[:self.batch_aq_num]
    #next_point_to_query = np.argmax(all_scores)
    #return all_indices[next_point_to_query]
 
  def random_aq(self, loader, model):
    random_indices = random.sample(range(len(loader.dataset)), self.batch_aq_num)
    return random_indices
