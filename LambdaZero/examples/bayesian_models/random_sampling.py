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
    A base class for all acquisition functions. All subclasses should
    override the `__call__` method.
    """
    # fixme
    # def random_selection(self, set_in, b): #set_in: input set; b: how many batch size we want to take
    #
    #     selection = np.random.choice(len(set_in), b) #take a random choice from 0 to len(set_in), do it for b times
    #     selected = set_in[[selection]]
    #     rest_selection = set_in[np.delete(range(len(set_in) - 1), selection)]
    #     set_in = TensorDataset(selected[0], selected[1])
    #     set_rest = TensorDataset(rest_selection[0], rest_selection[1])
    #
    #     return set_in, set_rest

    @abstractmethod
    def __call__(self, df, b):
        # fixme
        raise NotImplementedError("please, implement the call function")

class RandomAcquisition(AcquisitionFunction):

    def __call__(self, df, batch_size):
        # fixme (identation with 4 spaces)
        # fixme batch_size instead of b
      """

      :param df:
      :param b:
      :return:
      """
      x = torch.from_numpy(np.vstack(df['xs'].values).astype(np.float32))
      y = torch.Tensor(list(df['preds'].values))
      set_in = TensorDataset(x, y)
      return self.random_selection(set_in, b)

# class GreedyAcquisition(AcquisitionFunction):
#
#     def __call__(self, df, b):
#       df = df.sort_values('preds', ascending = False)
#       x = torch.from_numpy(np.vstack(df['xs'].values).astype(np.float32))
#       y = torch.Tensor(list(df['preds'].values))
#       x_in = x[:b]
#       x_rest = x[b:]
#       y_in = y[:b]
#       y_rest = y[b:]
#       set_in = TensorDataset(x_in, y_in)
#       set_rest = TensorDataset(x_rest, y_rest)
#       return set_in, set_rest

class EGreedyAcquisition(AcquisitionFunction):

    def __call__(self, df, b, noise_std, noise_mean):
        # fixme std, mean
      dist = torch.randn(len(df['preds'].values)) * cfg.std + cfg.mean
      df['preds'] = df['preds'].values + dist.detach().cpu().numpy() # fixme: use argsort(y)
      df = df.sort_values('preds', ascending = False)
      x = torch.from_numpy(np.vstack(df['xs'].values).astype(np.float32))
      y = torch.Tensor(list(df['preds'].values))

      x_in = x[:b]
      x_rest = x[b:]
      y_in = y[:b]
      y_rest = y[b:]

      set_in = TensorDataset(x_in, y_in)
      set_rest = TensorDataset(x_rest, y_rest)

      #return set_in, set_rest
      # return (idxs)

class cfg:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = 'd4_250k_clust.parquet'
    batch_size = 500
    epochs = 25
    std = 0.5
    mean = 0.
    max_len = 8000
    #acquisition_mode = #'Greedy' #Random, Greedy, EGreedy
    # fixme
    acquisition_func = EGreedyAcquisition

    #acquisition_map = {'Random':RandomAcquisition(), 'Greedy': GreedyAcquisition(), 'EGreedy': EGreedyAcquisition()}
    #acquisition_fxn = acquisition_map[acquisition_mode]

class CustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data)) 
        self.x = torch.stack(transposed_data[0])
        self.y = torch.stack(transposed_data[1])

# class utils:
#   def collate_wrapper(batch):
#       return CustomBatch(batch)
#
#   def get_top_k(df, k):
#       df = df.sort_values('ys', ascending = True)
#       y = df['ys'].values
#       preds = df['preds'].values
#       return preds[:k], y[:k]

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
      train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, collate_fn = utils.collate_wrapper)
      return train_dataset, train_loader

  def test_set(self):
      test_dataset = TensorDataset(self.x_test, self.y_test)
      test_loader = DataLoader(test_dataset, batch_size = cfg.batch_size, collate_fn = utils.collate_wrapper)
      return test_dataset, test_loader

  def val_set(self):
      val_dataset = TensorDataset(self.x_val, self.y_val)
      val_loader = DataLoader(val_dataset, batch_size = cfg.batch_size, collate_fn = utils.collate_wrapper)
      return val_dataset, val_loader



# todo: Trainer(dataset) # train/test/stop when needed/ --- is shared for the whole repo under utils
# todo: Aquirer (Trainer, acquisition function) # does aquisitons


class Trainer:
  def __init__(self, dataset, model = None, acquisition_func = None):
      # fixme: dataset
      # fixme: model should be passed as class
      # model(x)

    # fixme
    def _top_k(x):
        return k


    if model == None:
      self.model = torch.nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    #else:
    #  self.model = model
    #if acquisition_fxn == None:
    #  self.acquisition_fxn = RandomAcquisition()
    #else:
    #  self.acquisition_fxn = acquisition_fxn

  def train(self):
    fpd = FingerprintDataset()
    fpd.generate_fingerprint_dataset(cfg.data)
    model = self.model
    acquisition_fxn = self.acquisition_fxn
    train_set, train_loader = fpd.train_set()
    test_set, test_loader = fpd.test_set()
    optimizer = torch.optim.Adam(model.parameters())
    losses = [] # record loss on the train set
    test_losses = []
    top_mol = []
    top_y = []
    
    for index in tqdm(range(len(train_set) // cfg.batch_size)):
        if index == 0:
            set_in, set_rest = acquisition_fxn.random_selection(train_set, cfg.batch_size)
        train_loader = DataLoader(set_in, batch_size = cfg.batch_size, collate_fn = utils.collate_wrapper)
        model.train()
        
        for _ in range(cfg.epochs):
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pred = model(batch.x)
                loss = F.mse_loss(pred.squeeze(), batch.y)
                loss.backward()
                optimizer.step()

        test_mse, _ = self.test_epoch(test_loader, model)
        test_losses.append(test_mse)
        losses.append(loss)
        rest_loader = DataLoader(set_rest, batch_size = cfg.batch_size, collate_fn = utils.collate_wrapper)
        model.eval()
        preds = []
        ys = []
        xs = []
        
        for b in rest_loader:
          pred = model(b.x).detach().cpu().numpy().squeeze()
          preds.append(pred)
          xs.append(b.x)
          ys.append(b.y)

        xs = np.concatenate(xs, axis = 0)
        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=0)
        df = pd.DataFrame({'xs': list(xs), 'ys':ys, 'preds':preds})
        top_pred, top_val = utils.get_top_k(df, 1)
        top_mol.append(top_pred) 
        top_y.append(top_val)

        new_set_in, set_rest = acquisition_fxn(df, cfg.batch_size)
        set_in = torch.utils.data.ConcatDataset([set_in, new_set_in])

        if len(set_in) > cfg.max_len:
          break

    self.losses = losses
    self.test_losses = test_losses
    self.top_mol = top_mol
    self.top_y = top_y
    self.model = model

  def test_epoch(self, loader, model):
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
    return mse, test_out

  def generate_plot(self):
    plt.plot(np.linspace(0, len(self.test_losses) * cfg.batch_size, len(self.test_losses)), self.test_losses, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
    plt.plot(np.linspace(0, len(self.losses) * cfg.batch_size, len(self.losses)), self.losses, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title(cfg.acquisition_mode)
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()

  
if __name__ == '__main__':
    trainer = Trainer(acquisition_fxn = cfg.acquisition_fxn)
    trainer.train()
    trainer.generate_plot()