import numpy as np
import os, time, os.path as osp
import pandas as pd
import random
from matplotlib import pyplot as plt
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
from torch.utils.tensorboard import SummaryWriter

class FingerprintDataset:
    def __init__(self, config, device, filename='d4_250k_clust.parquet',x_name='fingerprint', y_name='dockscore'):
        self.batch_size = config['aq_batch_size']

        with open(osp.join('/home/jchen1/joanna-temp',filename), 'rb') as f:
          data = pd.read_parquet(f, engine='pyarrow')
 
        # takes the value from fingerprint column, stack vertically and turn to a tensor
        x_data = torch.from_numpy(np.vstack(data[x_name].values).astype(np.float32))
        y_data = torch.Tensor(list(data[y_name].values))
        x_data = x_data.to(device)
        y_data = y_data.to(device)
 
        # split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size=0.2)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2)
 
    def train_set(self):
        train_dataset = TensorDataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)#, collate_fn=collate_wrapper)
        return train_dataset, train_loader
 
    def test_set(self):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)#, collate_fn=collate_wrapper)
        return test_dataset, test_loader
 
    def val_set(self):
        val_dataset = TensorDataset(self.x_val, self.y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)#, collate_fn=collate_wrapper)
        return val_dataset, val_loader
 
class main(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = self.device

        # acquisition function class
        self.acquirer = Acquirer(config, device)
        self.writer = SummaryWriter('logs/')

        # load and spilt the data
        dataset = FingerprintDataset(config, device)
        self.train_set, _ = dataset.train_set()
        _, self.test_loader = dataset.test_set()

        self.top_mol_dict = {}
        self.top_batch_dict = {}

        self.regressor = ModelWithUncertainty(1024, 1)
        self.regressor = self.regressor.to(self.device)

        self.trainer = Trainer(self.config, torch.nn.MSELoss(), torch.optim.Adam(self.regressor.parameters(), lr=self.config["lr"]))

        self.is_aquired = np.zeros(len(self.train_set),dtype=np.bool)
        self.is_aquired[np.random.choice(np.arange(len(self.train_set)), self.config['aq_batch_size'])] = True

        self.aq_fxns = {'random': self.acquirer.random_aq, 'greedy': self.acquirer.egreedy_aq, 'egreedy': self.acquirer.egreedy_aq, 'ucb': self.acquirer.ucb_aq, 'thompson': self.acquirer.thompson_aq}
        self.aq_fxn = self.aq_fxns[config['aq_function']]
        # make epochs
        self.train_epoch = config["train_epoch"]

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
        for i in range(self.train_epoch):
            trn_loss = self.trainer.train(self.regressor, aq_loader)

        tst_loss = self.trainer.evalu(self.test_loader, self.regressor)
        test_losses.append(tst_loss)
        train_losses.append(trn_loss)

        # acquire more data
        rest_set = Subset(self.train_set, np.where(~self.is_aquired)[0])
        rest_loader = DataLoader(rest_set, batch_size=self.config['aq_batch_size'])

        ys, output = self.trainer.rest_evalu(aq_loader, self.regressor) 
        output_indices = np.argsort(ys) 
        ys = ys[output_indices]
        top_pred = ys[0]
        top_mol.append(top_pred)

        aq_indices = self.aq_fxn(rest_loader, self.regressor)
        self.is_aquired[aq_indices] = True

        batch_set = Subset(self.train_set, aq_indices)      
        batch_ys = [y.detach().cpu().numpy() for (_, y) in batch_set]
        top_batch.append(np.amin(np.array(batch_ys)))

        return {'train_top_mol_{}'.format(name) : top_pred, 'train_top_batch_{}'.format(name) : np.amin(np.array(batch_ys))}

        #for name, vals in self.top_mol_dict.items():
        #    plt.scatter(np.linspace(self.config['aq_batch_size'], self.config['sample_size'], len(vals)), vals, label = name)
        
        #plt.title('Best Molecules So Far')
        #plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
        #plt.xlabel("num_samples ")
        #plt.ylabel("best_energy_found")
        #plt.savefig('best_so_far.png')
        #plt.show()

        #for name, vals in self.top_batch_dict.items():
        #    plt.scatter(np.linspace(self.config['aq_batch_size'], self.config['sample_size'], len(vals)), vals, label = name)

        #plt.title('Best Molecules on Current Batch')
        #plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
        #plt.xlabel("num_samples ")
        #plt.ylabel("best_energy_found")
        #plt.savefig('best_on_batch.png')
        #plt.show()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.regressor.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.regressor.load_state_dict(torch.load(checkpoint_path))

class Trainer:
  def __init__(self, config, criterion, optimizer):
    self.criterion = criterion
    self.optimizer = optimizer
    self.config = config

  def train(self, regressor, train_loader):
  
      for i, batch in enumerate(train_loader):
          self.optimizer.zero_grad()
          output = regressor(batch[0])
          loss = self.criterion(output.squeeze(), batch[1].squeeze())
          ys = batch[1].squeeze()
          loss.backward()
          self.optimizer.step()

      return loss
  
  def rest_evalu(self, rest_loader, regressor):
      if self.config['mcdrop']:
          regressor.train()
      else:
          regressor.eval()
      y_all = []
      out_all = []
      for i, (x, y) in enumerate(rest_loader):
          output = regressor(x)
          y_all.append(y.detach().cpu().numpy())
          out_all.append(1)
  
      y_all = np.concatenate(y_all, axis = 0)
      return y_all, out_all
  
  def evalu(self, test_loader, regressor):
      mean_loss = []
      for i, batch in enumerate(test_loader):
          output = regressor(batch[0])
          loss = self.criterion(output.squeeze(), batch[1].squeeze())
  
          mean_loss.append(loss.detach().cpu().numpy())
      mean_loss = np.array(mean_loss)
      print('Test mean loss: {}'.format(np.mean(mean_loss)))
      self.mean_loss = mean_loss
      return mean_loss
 
class ModelWithUncertainty(nn.Module):
    """
    This can be Bayesian Linear Regression, Gaussian Process, Ensemble etc...
    Also note before I have been thinking that these models have additional feature extractor, which acts on X
    first, this is still sensible to have here if we think necessary. eg this may be MPNN on SMILES
    """
    # Bayesian Linear Regression
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.blinear1 = BayesianLinear(input_dim, 512)
 
        self.blinear2 = BayesianLinear(512, output_dim)
 
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = self.relu(x_)
        x_ = self.dropout(x_)
        return self.blinear2(x_)
 
    def fit(self, aq_loader, test_loader):
        pass
 
    def update(self, x_train, y_train):
        """
        Some models allow you to update them more cheaply than retraining with all
        points (eg low rank update for Bayesian regression.) Hence you have an update
        function too.
        """
        self.model = self.model.condition_on_observations(x_train, y_train)
 
    def get_predict_mean_and_variance_func(self):
        """
        returns a function
        that should take in X_test [N, d] and return a two element tuple of tensor ([N,f], [N,f])
        which contains the predictive mean and variance at that output point
        f is the dimensionality of y, which is probably always going to be 1...?
        N is the batch size
        """
        def prediction(x_test):
            preds = self.forward(x_test)
            mean_x = np.mean(preds.detach().cpu().numpy())
            var_x = np.var(preds.detach().cpu().numpy())
 
            return var_x, mean_x
 
        return prediction
 
    def get_predict_sample(self):
        """
        This samples from your distribution over models/parameters/hyperparameters and returns
        an evaluator that tells you what the sampled model will predict at that point. Useful for
        eg Thompson sampling.
        returns a function
        that should take in X_test [N, d] and return a tensor [N,f]
        that is the predicted value at that point given the model parameters you have sampled
        f is the dimensionality of y, which is probably always going to be 1...?
        N is the batch size, d the dimensionality of features
        """
 
        def evaluator(x_test):
            out = self.forward(x_test)
            return out
 
        return evaluator

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
    all_batch = []
    all_indices = []
    j = 0
    for i, (x, y) in enumerate(loader):
      # ^ this is the loop which you parallelize with eg Ray
      all_batch.append(x)
      mean_on_batch, var_on_batch = mu_var(x)
      score_on_batch = mean_on_batch + self.kappa * var_on_batch
      all_scores.append(score_on_batch)
      all_indices.append(range(j * self.batch_aq_num, (j+1) * self.batch_aq_num))
      j += 1
    next_point_to_query = np.argmax(all_scores)
    return all_indices[next_point_to_query]
 
  def random_aq(self, loader, model):
    random_indices = random.sample(range(len(loader.dataset)), self.batch_aq_num)
    return random_indices

aq_functions = ['random', 'greedy', 'egreedy', 'ucb', 'thompson']

cfg = {
  "trainer": main,
   "trainer_config": {
        "aq_function": aq_functions[0],
        "lr": 0.001,
        "train_epoch": 3,
        "sample_size": 20000,
        "aq_batch_size": 512,
        "mcdrop": False,
	"summaries_dir": '/home/jchen1/joanna-temp/summaries'
        },
  
  "memory": 20 * 10 ** 9,
  "stop": {"training_iteration": 100},
  }

config = cfg
  
if __name__ == '__main__':
    ray.init()

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop=config["stop"], #EarlyStop(),
                        resources_per_trial={
                           "cpu": 2, 
                           "gpu": 1.0
                        },
                        num_samples=100,
                        checkpoint_at_end=False,
			local_dir='/home/jchen1/joanna-temp/summaries',
                        checkpoint_freq=100000)