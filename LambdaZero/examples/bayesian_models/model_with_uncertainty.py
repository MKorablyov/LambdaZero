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
from torch.utils.tensorboard import SummaryWriter
import sklearn
from sklearn.model_selection import train_test_split

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device: {}'.format(device))

class cfg:
  sample_size = 50000
  batch_size = 500
 
class FingerprintDataset:
    def __init__(self, filename='d4_250k_clust.parquet', batch_size=cfg.batch_size, x_name='fingerprint', y_name='dockscore'):
        data = pd.read_parquet(filename, engine='pyarrow')
 
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
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size)
        return train_dataset, train_loader
 
    def test_set(self):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
        return test_dataset, test_loader
 
    def val_set(self):
        val_dataset = TensorDataset(self.x_val, self.y_val)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
        return val_dataset, val_loader
 
def main():
    regressor = ModelWithUncertainty(1024, 1)
    regressor = regressor.to(device)

    acquirer = Acquirer(cfg.batch_size, device)
    trainer = Trainer(torch.nn.MSELoss(), torch.optim.Adam(regressor.parameters(), lr=0.001))
 
    dataset = FingerprintDataset()
    train_set, train_loader = dataset.train_set()
    test_set, test_loader = dataset.test_set()
 
    is_aquired = np.zeros(len(train_set),dtype=np.bool)
    is_aquired[np.random.choice(np.arange(len(train_set)), cfg.batch_size)] = True
 
    test_losses = []
    train_losses = []
 
    top_mol = []
    top_y = []
    top_batch = []

    while np.count_nonzero(is_aquired) < len(train_set) and np.count_nonzero(is_aquired) < cfg.sample_size:
        aq_set = Subset(train_set, np.where(is_aquired)[0])
        aq_loader = torch.utils.data.DataLoader(aq_set, batch_size=cfg.batch_size)
        
        # train model
        for i in range(3):
            trn_loss = trainer.train(regressor, aq_loader)
 
        tst_loss = trainer.evalu(test_loader, regressor)
 
        test_losses.append(tst_loss)
        train_losses.append(trn_loss)
 
        # acquire more data
        rest_set = Subset(train_set, np.where(~is_aquired)[0])
        rest_loader = DataLoader(rest_set, batch_size=cfg.batch_size)

        ys, output = trainer.rest_evalu(aq_loader, regressor) 
        output_indices = np.argsort(ys)
        ys = ys[output_indices]
        top_pred = ys[0]
        top_mol.append(top_pred)

        writer.add_scalar('Values', top_pred)

        # randomly acquire batch size values from the rest_set and take the indicies
        aq_indices = acquirer.random_aq(rest_loader, regressor)
        # return a indices list, make sure not acquire the thing we have already acquire
        batch_aq = np.arange(len(train_set))[~is_aquired][aq_indices] 

        is_aquired[batch_aq] = True 

        #*** Bug here! should use batch_aq rather than aq_indices
        batch_set = Subset(train_set, aq_indices) # return x, y in the original dataset     

        batch_ys = [y.detach().cpu().numpy() for (_, y) in batch_set]
        top_batch.append(np.amin(np.array(batch_ys)))


    writer.close()

    plt.scatter(np.linspace(cfg.batch_size, cfg.sample_size, len(top_mol)), top_mol, label = 'Random')
    plt.title('Best Molecules So Far')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("num_samples ")
    plt.ylabel("best_energy_found")
    plt.savefig('best_so_far.png')
    plt.show()

    plt.scatter(np.linspace(cfg.batch_size, cfg.sample_size, len(top_batch)), top_batch, label = 'Random')
    plt.title('Best Molecules on Current Batch')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("num_samples ")
    plt.ylabel("best_energy_found")
    plt.savefig('best_on_batch.png')
    plt.show()
 
 
class Trainer:
  def __init__(self, criterion, optimizer):
    self.criterion = criterion
    self.optimizer = optimizer

  def train(self, regressor, train_loader):
      for i, batch in enumerate(train_loader):
          self.optimizer.zero_grad()
          output = regressor(batch[0])
          loss = self.criterion(output.squeeze(), batch[1].squeeze())
          ys = batch[1].squeeze() #batch[0] is x and batch[1] is y
          loss.backward()
          self.optimizer.step()
      return loss
  
  def rest_evalu(self, rest_loader, regressor):
      regressor = regressor.eval()
      y_all = []
      out_all = []

      for i, (x, y) in enumerate(rest_loader):
          output = regressor(x)
          y_all.append(y.detach().cpu().numpy())
          out_all.append(output.squeeze().detach().cpu().numpy())
  
      y_all = np.concatenate(y_all, axis = 0) # real value
      out_all = np.concatenate(out_all, axis = 0) # predicted
  
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
        self.blinear1 = torch.nn.Linear(input_dim, 512)
        self.blinear2 = torch.nn.Linear(512, output_dim)
 
    def forward(self, x):
        x_ = self.blinear1(x)
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
  def __init__(self, batch_aq_num, device, noise_std = 1, noise_mean = 0):
    self.batch_aq_num = batch_aq_num #batch size
    self.device = device
    self.kappa = 0.2
    self.noise_std = noise_std
    self.noise_mean = noise_mean

  def egreedy_aq(self,loader, model):
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
  
if __name__ == '__main__':
    main()

