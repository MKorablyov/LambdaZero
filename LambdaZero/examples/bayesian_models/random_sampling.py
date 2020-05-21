import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from abc import ABC, abstractmethod

class AcquisitionFunction(ABC):
    """
    A base class for all acquisition functions. All subclasses should override the `__call__` method.
    """

    def random_selection(self, set_in, batch_size):
        """
        usage: grab the initial batch size b to train on, and also used through out Random acquisition 
                function, returns two tensor datasets for the selected rows and the rest.
        parameters: 
                set_in = input set
                batch_size = batch size
        return:
                set_in = batch-size randomly selected tensor dataset
                set_rest = (length of input set - batch size) remainly unselected tensor dataset
        """
        selection = np.random.choice(len(set_in), batch_size) # take a random choice from 0 to len(set_in), do it for b times, return the index
        selected = set_in[[selection]]
        rest_selection = set_in[np.delete(range(len(set_in) - 1), selection)]
        set_in = TensorDataset(selected[0], selected[1]) # return x (fingerprint) and y (dockscore)
        set_rest = TensorDataset(rest_selection[0], rest_selection[1])

        return set_in, set_rest

    @abstractmethod
    def __call__(self, df, b):
        raise NotImplementedError("please, implement the call function")

class RandomAcquisition(AcquisitionFunction):

    def __call__(self, df, batch_size):
      """
       usage: perform the Random acquisition function
       parameters: 
               df = dataframe of dockscore prediction (y_hat) and real dockscore values (y)
                    (we put a dataframe into the acquisition function because it is easy to sort and extract from)
               batch_size = batch size, the number we want to grab from the acquisition function
       return: batch-size randomly selected tensor dataset & (length of input set - batch size) remainly 
               unselected tensor dataset using random acquisition function
       """
      x = torch.from_numpy(np.vstack(df['xs'].values).astype(np.float32))
      y = torch.Tensor(list(df['preds'].values))
      set_in = TensorDataset(x, y)
      return self.random_selection(set_in, batch_size)

class GreedyAcquisition(AcquisitionFunction):

    def __call__(self, df, batch_size, noise_std = 0., noise_mean = 0.):
      """
      usage: perform the Greedy and e-greedy acquisition function
      parameters: 
              df = dataframe of dockscore prediction (y_hat) and real dockscore values (y)   
              batch_size = batch size, the number we want to grab from the acquisition function
              noise_std = standard deviation of noise for e-greedy
              noise_mean = mean of noise for e-greedy
              *Note: when noise_std = noise_mean = 0, e-greedy = greedy
      return: 
            set_in: batch-size randomly selected tensor dataset using greedy/e-greedy acquisition function
            set_rest:(length of input set - batch size) remainly unselected tensor dataset using greedy/e-greedy acquisition function
      """
      dist = torch.randn(len(df['preds'].values)) * noise_std + noise_mean
      df['preds'] = df['preds'].values + dist.detach().cpu().numpy()
      df = df.sort_values('preds', ascending = False)
      x = torch.from_numpy(np.vstack(df['xs'].values).astype(np.float32))
      y = torch.Tensor(list(df['preds'].values))

      x_in = x[:batch_size]
      x_rest = x[batch_size:]
      y_in = y[:batch_size]
      y_rest = y[batch_size:]

      set_in = TensorDataset(x_in, y_in)
      set_rest = TensorDataset(x_rest, y_rest)

      return set_in, set_rest

class cfg:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = 'd4_250k_clust.parquet'
    batch_size = 500
    epochs = 10
    classic_epochs = 100
    max_len = 8000
    acquisition_mode = 'Greedy' # Random, Greedy
    acquisition_map = {'Random':RandomAcquisition(), 'Greedy': GreedyAcquisition()}
    acquisition_fxn = acquisition_map[acquisition_mode]

class CustomBatch:
    """
    Name the data batch x (fingerprint) and y (dockscore) to make it easier to manage
    """
    def __init__(self, data):
        transposed_data = list(zip(*data)) 
        self.x = torch.stack(transposed_data[0])
        self.y = torch.stack(transposed_data[1])

def collate_wrapper(batch):
    """
    Wrap the data up so we are able to send to CustomBatch
    """
    return CustomBatch(batch)

class FingerprintDataset:
  def __init__(self):
      pass

  def generate_fingerprint_dataset(self, filename = 'd4_250k_clust.parquet', batch_size = 500, x_name = 'fingerprint', y_name = 'dockscore'):
      data = pd.read_parquet(filename, engine='pyarrow')

      # remove the 20% of worst energy
      data = data.sort_values(y_name)[:-int(round(len(data)*0.2))]
      
      # takes the value from fingerprint column, stack vertically and turn to a tensor
      x_data = torch.from_numpy(np.vstack(data[x_name].values).astype(np.float32))
      y_data = torch.Tensor(list(data[y_name].values)) 
      
      # split the data
      self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size = 0.2)
      self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size = 0.2)

  def train_set(self):
      train_dataset = TensorDataset(self.x_train, self.y_train)
      train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, collate_fn = collate_wrapper)
      return train_dataset, train_loader

  def test_set(self):
      test_dataset = TensorDataset(self.x_test, self.y_test)
      test_loader = DataLoader(test_dataset, batch_size = cfg.batch_size, collate_fn = collate_wrapper)
      return test_dataset, test_loader

  def val_set(self):
      val_dataset = TensorDataset(self.x_val, self.y_val)
      val_loader = DataLoader(val_dataset, batch_size = cfg.batch_size, collate_fn = collate_wrapper)
      return val_dataset, val_loader

class Trainer:
  def __init__(self, dataset, model = None, acquisition_fxn = None):
    if model == None:
      self.model = torch.nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    else:
      self.model = model
    self.model_classic = torch.nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    if acquisition_fxn == None:
      self.acquisition_fxn = RandomAcquisition()
    else:
      self.acquisition_fxn = acquisition_fxn
    self.dataset = dataset
    
  def _get_top_k(self, df, k):
    """
      usage: get the top k dockscore
      parameters: 
              df = dataframe of dockscore prediction (y_hat) and real dockscore values (y)
              k = how many top score we want to get
      return: the value of the top k prediction values and the real values
    """
    df = df.sort_values('ys', ascending = True)
    y = df['ys'].values
    preds = df['preds'].values
    classic_preds = df['classic'].values
    return preds[:k], y[:k], classic_preds[:k]

  def train_classic(self):
    """
      usage: trains the model without any acquisition function
      return: sets the losses, test_losses, top molecule, and top y values lists for plotting
    """
    fpd = self.dataset
    fpd.generate_fingerprint_dataset(cfg.data)
    model = self.model_classic
    train_set, train_loader = fpd.train_set()
    test_set, test_loader = fpd.test_set()
    optimizer = torch.optim.Adam(model.parameters())
    losses = [] # record loss on the train set
    test_losses = []

    model.train()
    for _ in tqdm(range(cfg.classic_epochs)):
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(batch.x)
            loss = F.mse_loss(pred.squeeze(), batch.y)
            loss.backward()
            optimizer.step()

        test_mse, _, _ = self.test_epoch(test_loader, model)
        test_losses.append(test_mse)
        losses.append(loss)

    self.losses_classic = losses
    self.test_losses_classic = test_losses
    self.model_classic = model

  def train(self):
    """
      usage: trains the model using the selected acquisition funtion, logs the losses, test losses, retrieves the top molecules and their real-valued counterpart
      return: sets the losses, test_losses, top molecule, and top y values lists for plotting
    """
    fpd = self.dataset
    fpd.generate_fingerprint_dataset(cfg.data)
    model = self.model
    acquisition_fxn = self.acquisition_fxn
    train_set, train_loader = fpd.train_set()
    test_set, test_loader = fpd.test_set()
    optimizer = torch.optim.Adam(model.parameters())
    losses = []
    test_losses = []
    top_mol = []
    top_y = []
    classic_top_mol = []
    
    for index in tqdm(range(len(train_set) // cfg.batch_size)):
        if index == 0:
            set_in, set_rest = acquisition_fxn.random_selection(train_set, cfg.batch_size)
        train_loader = DataLoader(set_in, batch_size = cfg.batch_size, collate_fn = collate_wrapper)
        model.train()
        
        for _ in range(cfg.epochs):
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pred = model(batch.x)
                loss = F.mse_loss(pred.squeeze(), batch.y)
                loss.backward()
                optimizer.step()

        test_mse, _, _ = self.test_epoch(test_loader, model)
        test_losses.append(test_mse)
        losses.append(loss)
        rest_loader = DataLoader(set_rest, batch_size = cfg.batch_size, collate_fn = collate_wrapper)
        model.eval()
        preds = []
        classic_preds = []
        ys = []
        xs = []
        
        for batch in rest_loader:
          classic_pred = self.model_classic(batch.x).detach().cpu().numpy().squeeze()
          classic_preds.append(classic_pred)
          pred = model(batch.x).detach().cpu().numpy().squeeze()
          preds.append(pred)
          xs.append(batch.x)
          ys.append(batch.y)

        xs = np.concatenate(xs, axis = 0)
        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=0)
        classic_preds = np.concatenate(classic_preds, axis=0)
        df = pd.DataFrame({'xs': list(xs), 'ys':ys, 'preds':preds, 'classic': classic_preds})
        top_pred, top_val, classic_top_pred = self._get_top_k(df, k = 1)
        top_mol.append(top_pred) 
        top_y.append(top_val)
        classic_top_mol.append(classic_top_pred)

        new_set_in, set_rest = acquisition_fxn(df, cfg.batch_size)
        set_in = torch.utils.data.ConcatDataset([set_in, new_set_in])

        if len(set_in) > cfg.max_len:
          break

    self.losses = losses
    self.test_losses = test_losses
    self.top_mol = top_mol
    self.top_y = top_y
    self.classic_top_mol = classic_top_mol
    self.model = model

  def test_epoch(self, loader, model):
    """
      usage: evaluates the model on the test set
      parameters:
        loader = dataloader with all the values on the test set
        model = input model
      return: 
        mse = mse between the predicted scores and test scores
        test_out = dataframe of the fingerprint, actual dockscore values and predicted dockscore values
    """
    model.eval()
    xs = []
    ys = []
    preds = []
    
    for data in loader:
        xs.append(data.x.detach().cpu().numpy())
        ys.append(data.y.detach().cpu().numpy())
        preds.append(model(data.x).detach().cpu().numpy())

    xs = np.concatenate(xs, axis = 0)
    ys = np.concatenate(ys, axis=0)
    preds = np.concatenate(preds, axis=0)
    mse = F.mse_loss(torch.from_numpy(preds.squeeze()).to(cfg.device), torch.from_numpy(ys.squeeze()).to(cfg.device))
    test_out = pd.DataFrame({"xs":list(xs), "ys": ys.squeeze(), "preds": preds.squeeze()})
    return mse, _, test_out

  def generate_plot(self):
    """
      usage: generate MSE plot from the train set and the test set for user-defined acquisition function
    """
    plt.plot(np.linspace(0, len(self.test_losses) * cfg.batch_size, len(self.test_losses)), self.test_losses, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
    plt.plot(np.linspace(0, len(self.losses) * cfg.batch_size, len(self.losses)), self.losses, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title(cfg.acquisition_mode)
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()

  def dot_plot(self):
    """
      usage: generate plots of models w/ acquisition function predicted on the top molecules comparing to model trained w/o acquisition function and the actual docking score. 
    """
    plt.scatter(np.linspace(0, len(self.top_y) * cfg.batch_size, len(self.top_y)), self.top_mol, label = cfg.acquisition_mode)
    plt.scatter(np.linspace(0, len(self.top_y) * cfg.batch_size, len(self.top_y)), self.top_y, label = 'Actual')
    plt.scatter(np.linspace(0, len(self.top_y) * cfg.batch_size, len(self.top_y)), self.classic_top_mol, label = 'w/o acquisition')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel('Search Space')
    plt.ylabel('Score')
    plt.show()
  
if __name__ == '__main__': 
    dataset = FingerprintDataset()
    trainer = Trainer(dataset = dataset, acquisition_fxn = cfg.acquisition_fxn)
    trainer.train_classic()
    trainer.train()
    trainer.dot_plot()

