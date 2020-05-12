class EarlyStopping:
# Taken from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        th.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

import pandas as pd
import numpy as np
import torch as th
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F

device = 'cuda:0' if th.cuda.is_available() else 'cpu'

class cfg:
    # data could be found here: https://github.com/MKorablyov/brutal_dock/tree/master/d4/raw
    data = 'd4_250k_clust.parquet'#"/home/maksym/Datasets/brutal_dock/d4/raw/d4_100k_clust.parquet"
    batch_size = 500
    epochs = 6
    std = 1.
    mean = 0.
    patience = 10
    mode = 'y'
    # k = 10

#sns.distplot(df["dockscore"])
#plt.show()

# total_budget = 3000

# strategies
# S1: greedy
# batch_size = 500, num_interations = 6

def train_random(x, y, model, optimizer, x_test, y_test):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    test_mse = []
    bound = 0
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    top_mol = []
    top_y = []
    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True)

    for i in tqdm(range(0, len(x), batch_size)):
        if bound == 0:
            bound = i + batch_size
            x_in = x[0:bound]
            x_in = x_in.to(device)
            y_in = y[0:bound]
            y_in = y_in.to(device)

            x_rest = x[bound:]
            x_rest = x_rest.to(device)
            y_rest = y[bound:]
            y_rest = y_rest.to(device)     

        x_in = x_in.to(device)
        y_in = y_in.to(device)
        x_rest = x_rest.to(device)
        y_rest = y_rest.to(device)

        model.train()
        assert model.training

        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            pred = model(x_in)
            loss = F.mse_loss(pred.squeeze(1), y_in)
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        loss_all += loss.item()
        losses.append(loss)

        mse, _, _, _ = test_epoch(x_test, y_test, model)
        test_mse.append(mse)

        early_stopping(mse, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model.eval()
        preds = model(x_rest)
        df = pd.DataFrame({'x': list(x_rest.detach().cpu().numpy()), 'y':list(y_rest.detach().cpu().numpy()), 'preds':list(preds.detach().cpu().numpy().squeeze())})
        df = df.sample(frac=1)

        x_rest = np.vstack(df['x'].values)
        y_rest = np.vstack(df['preds'].values).squeeze()
        
        df['differences'] = df['y'] - df['preds']
        df = df.sort_values('differences')

        top_pred, top_val = get_top_k(df, 1, cfg.mode)
        top_mol.append(top_pred)
        top_y.append(top_val)

        x_in = x_in.detach().cpu().numpy()
        y_in = y_in.detach().cpu().numpy()

        x_in = np.concatenate((x_in, x_rest[:cfg.batch_size]))
        y_in = np.concatenate((y_in, y_rest[:cfg.batch_size]))
        x_rest = x_rest[cfg.batch_size:]
        y_rest = y_rest[cfg.batch_size:]

        in_tmp = np.c_[x_in.reshape(len(x_in), -1), y_in.reshape(len(y_in), -1)]
        rest_tmp = np.c_[x_rest.reshape(len(x_rest), -1), y_rest.reshape(len(y_rest), -1)]

        np.random.shuffle(in_tmp)
        np.random.shuffle(rest_tmp)
        
        x_in = in_tmp[:, :x_in.size//len(x_in)].reshape(x_in.shape)
        y_in = in_tmp[:, x_in.size//len(x_in):].reshape(y_in.shape)
        x_rest = rest_tmp[:, :x_rest.size//len(x_rest)].reshape(x_rest.shape)
        y_rest = rest_tmp[:, x_rest.size//len(x_rest):].reshape(y_rest.shape)

        x_in = th.from_numpy(x_in).float().to(device)
        y_in = th.from_numpy(y_in).float().to(device)
        x_rest = th.from_numpy(x_rest).float().to(device)
        y_rest = th.from_numpy(y_rest).float().to(device)

    return loss_all / len(x), losses, test_mse, top_mol, top_y

def train_greedy(x, y, model, optimizer, x_test, y_test):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    test_mse = []
    bound = 0
    top_mol = []
    top_y = []
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True)

    for i in tqdm(range(0, len(x), batch_size)):
        if bound == 0:
            bound = i + batch_size
            x_in = x[0:bound]
            x_in = x_in.to(device)
            y_in = y[0:bound]
            y_in = y_in.to(device)

            x_rest = x[bound:]
            x_rest = x_rest.to(device)
            y_rest = y[bound:]
            y_rest = y_rest.to(device)     

        x_in = x_in.to(device)
        y_in = y_in.to(device)
        x_rest = x_rest.to(device)
        y_rest = y_rest.to(device)

        model.train()
        assert model.training
        for _ in range(cfg.epochs):
            for j in range(0, len(x_in), batch_size):

                optimizer.zero_grad()
                pred = model(x_in[i:i+batch_size])
                loss = F.mse_loss(pred.squeeze(1), y_in[i:i+batch_size])
                loss.backward()
                optimizer.step()

        loss_all += loss.item()
        losses.append(loss)

        mse, _, _, _ = test_epoch(x_test, y_test, model)
        early_stopping(mse, model)
        test_mse.append(mse)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        model.eval()
        preds = model(x_rest)
        
        df = pd.DataFrame({'x': list(x_rest.detach().cpu().numpy()), 'y':list(y_rest.detach().cpu().numpy()), 'preds':list(preds.detach().cpu().numpy().squeeze())})
        df = df.sort_values('preds', ascending = False)

        x_rest = np.vstack(df['x'].values)
        y_rest = np.vstack(df['preds'].values).squeeze()
        
        df['differences'] = df['y'] - df['preds']
        df = df.sort_values('differences')

        top_pred, top_val = get_top_k(df, 1, cfg.mode)
        top_mol.append(top_pred)
        top_y.append(top_val)
        
        try:
            x_in = x_in.detach().cpu().numpy()
            y_in = y_in.detach().cpu().numpy()

            x_in = np.concatenate((x_in, x_rest[:cfg.batch_size]))
            y_in = np.concatenate((y_in, y_rest[:cfg.batch_size]))
            x_rest = x_rest[cfg.batch_size:]
            y_rest = y_rest[cfg.batch_size:]

            in_tmp = np.c_[x_in.reshape(len(x_in), -1), y_in.reshape(len(y_in), -1)]
            rest_tmp = np.c_[x_rest.reshape(len(x_rest), -1), y_rest.reshape(len(y_rest), -1)]

            np.random.shuffle(in_tmp)
            np.random.shuffle(rest_tmp)
            
            x_in = in_tmp[:, :x_in.size//len(x_in)].reshape(x_in.shape)
            y_in = in_tmp[:, x_in.size//len(x_in):].reshape(y_in.shape)
            x_rest = rest_tmp[:, :x_rest.size//len(x_rest)].reshape(x_rest.shape)
            y_rest = rest_tmp[:, x_rest.size//len(x_rest):].reshape(y_rest.shape)

            x_in = th.from_numpy(x_in).float().to(device)
            y_in = th.from_numpy(y_in).float().to(device)
            x_rest = th.from_numpy(x_rest).float().to(device)
            y_rest = th.from_numpy(y_rest).float().to(device)
         
        except:
            break

    return loss_all / len(x), losses, test_mse, top_mol, top_y

def train_greedy_error(x, y, model, optimizer, x_test, y_test):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    test_mse = []
    top_mol = []
    top_y = []
    bound = 0
    std = cfg.std
    mean = cfg.mean
    crit = th.nn.MSELoss()
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True)

    for i in tqdm(range(0, len(x), batch_size)):
        if bound == 0:
            bound = i + batch_size
            x_in = x[0:bound]
            x_in = x_in.to(device)
            y_in = y[0:bound]
            y_in = y_in.to(device)

            x_rest = x[bound:]
            x_rest = x_rest.to(device)
            y_rest = y[bound:]
            y_rest = y_rest.to(device)

        x_in = x_in.to(device)
        y_in = y_in.to(device)
        x_rest = x_rest.to(device)
        y_rest = y_rest.to(device)

        model.train()
        for _ in range(cfg.epochs):
            for j in range(0, len(x_in), batch_size):
                optimizer.zero_grad()
                pred = model(x_in[i:i+batch_size])
                loss = F.mse_loss(pred.squeeze(1), y_in[i:i+batch_size])
                loss.backward()
                optimizer.step()

        loss_all += loss.item()
        losses.append(loss)
        
        mse, _, _, _ = test_epoch(x_test, y_test, model)
        test_mse.append(mse)

        early_stopping(mse, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        preds = model(x_rest)
        dist = th.randn(preds.size()) * std + mean
        dist = dist.to(device)
        preds = preds + dist

        df = pd.DataFrame({'x': list(x_rest.detach().cpu().numpy()), 'y':list(y_rest.detach().cpu().numpy()), 'preds':list(preds.detach().cpu().numpy().squeeze())})
        df = df.sort_values('preds', ascending = False)

        x_rest = np.vstack(df['x'].values)
        y_rest = np.vstack(df['preds'].values).squeeze()
        
        df['differences'] = df['y'] - df['preds']
        df = df.sort_values('differences')

        top_pred, top_val = get_top_k(df, 1, cfg.mode)
        top_mol.append(top_pred)
        top_y.append(top_val)
        
        x_in = np.concatenate((x_in.detach().cpu().numpy(), x_rest[-cfg.batch_size:]))
        y_in = np.concatenate((y_in.detach().cpu().numpy(), y_rest[-cfg.batch_size:]))
        x_rest = x_rest[:-cfg.batch_size]
        y_rest = y_rest[:-cfg.batch_size]

        in_tmp = np.c_[x_in.reshape(len(x_in), -1), y_in.reshape(len(y_in), -1)]
        rest_tmp = np.c_[x_rest.reshape(len(x_rest), -1), y_rest.reshape(len(y_rest), -1)]
        
        np.random.shuffle(in_tmp)
        np.random.shuffle(rest_tmp)

        x_in = in_tmp[:, :x_in.size//len(x_in)].reshape(x_in.shape)
        y_in = in_tmp[:, x_in.size//len(x_in):].reshape(y_in.shape)
        x_rest = rest_tmp[:, :x_rest.size//len(x_rest)].reshape(x_rest.shape)
        y_rest = rest_tmp[:, x_rest.size//len(x_rest):].reshape(y_rest.shape)

        x_in = th.from_numpy(x_in).float().to(device)
        y_in = th.from_numpy(y_in).float().to(device)
        x_rest = th.from_numpy(x_rest).float().to(device)
        y_rest = th.from_numpy(y_rest).float().to(device)
        
    return loss_all / len(x), losses, test_mse, top_mol, top_y

def test_epoch(x, y, model):
    model.eval()
    ys = []
    preds = []
    batch_size = cfg.batch_size

    for i in range(0, len(x), batch_size):
        x_in = x[i:i+batch_size]
        x_in = x_in.to(device)
        y_in = y[i:i+batch_size]
        ys.append(y_in.detach().cpu().numpy())
        preds.append(model(x_in).detach().cpu().numpy())
        
    ys = np.concatenate(ys, axis=0)
    preds = np.concatenate(preds, axis=0)
    mse = np.mean((ys - preds)**2)
    mae = np.mean(np.abs(ys - preds))

    return mse, mae, ys, preds


class Model(th.nn.Module):
    def __init__(self, inp_size, output_size):
        super(Model, self).__init__()
        self.fc1 = th.nn.Linear(1024, 1536)
        self.tanh = th.nn.ReLU()
        self.fc2 = th.nn.Linear(1536, 512)
        self.tanh2 = th.nn.ReLU()
        self.fc3 = th.nn.Linear(512, output_size)

    def forward(self, x):
        fc = self.fc1(x)
        tanh = self.tanh(fc)
        tanh2 = self.tanh2(self.fc2(tanh))
        return self.fc3(tanh2)

def get_top_k(df, k, mode = 'difference'):
    if mode == 'y':
        df = df.sort_values('y', ascending = True)
    elif mode == 'difference':
        df = df.sort_values('difference')
    y = df['y'].values
    preds = df['preds'].values
    return preds[:k], y[:k]

df1 = pd.read_parquet(cfg.data)
df1 = df1.iloc[np.random.permutation(len(df1))]
x = th.from_numpy(np.vstack(df1['fingerprint'].values).astype(np.float32))
y = th.Tensor(list(df1['dockscore'].values))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

x_test = np.asarray([t.numpy() for t in x_test], dtype = np.float32)
y_test = np.asarray([t.numpy() for t in y_test], dtype = np.float32)
x_test = th.FloatTensor(x_test)
y_test = th.FloatTensor(y_test)

#RANDOM
model = Model(1024, 1)#(x.shape[0], 1)
model = model.to(device)

optimizer = th.optim.Adam(model.parameters())
    
x_train = np.asarray([t.numpy() for t in x_train], dtype = np.float32)
y_train = np.asarray([t.numpy() for t in y_train], dtype = np.float32)
x_train = th.FloatTensor(x_train)
y_train = th.FloatTensor(y_train)

avg_loss, train_mse, test_mse, top_mol_rand, top_y_rand = train_random(x_train, y_train, model, optimizer, x_test, y_test)
mse, mae, ys, preds = test_epoch(x_val, y_val, model)

plt.plot(np.linspace(0, len(test_mse) * cfg.batch_size, len(test_mse)), test_mse, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
plt.plot(np.linspace(0, len(train_mse) * cfg.batch_size, len(train_mse)), train_mse, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title('Random')
plt.xlabel("Sample Size")
plt.ylabel("MSE")
plt.show()

#GREEDY
model = Model(1024, 1)#(x.shape[0], 1)
model = model.to(device)
optimizer = th.optim.Adam(model.parameters())
    
x_train = np.asarray([t.numpy() for t in x_train], dtype = np.float32)
y_train = np.asarray([t.numpy() for t in y_train], dtype = np.float32)
x_train = th.FloatTensor(x_train)
y_train = th.FloatTensor(y_train)

avg_loss, train_mse, test_mse, top_mol_greedy, top_y_greedy = train_greedy(x_train, y_train, model, optimizer, x_test, y_test)
mse, mae, ys, preds = test_epoch(x_val, y_val, model)

plt.plot(np.linspace(0, len(test_mse) * cfg.batch_size, len(test_mse)), test_mse, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
plt.plot(np.linspace(0, len(train_mse) * cfg.batch_size, len(train_mse)), train_mse, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title('Greedy')
plt.xlabel("Sample Size")
plt.ylabel("MSE")
plt.show()

#EPSILON GREEDY
model = Model(1024, 1)#(x.shape[0], 1)
model = model.to(device)

optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
    
x_train = np.asarray([t.numpy() for t in x_train], dtype = np.float32)
y_train = np.asarray([t.numpy() for t in y_train], dtype = np.float32)
x_train = th.FloatTensor(x_train)
y_train = th.FloatTensor(y_train)

avg_loss, train_mse, test_mse, top_mol_eps, top_y_eps = train_greedy_error(x_train, y_train, model, optimizer, x_test, y_test)
mse, mae, ys, preds = test_epoch(x_val, y_val, model)

plt.plot(np.linspace(0, len(test_mse) * cfg.batch_size, len(test_mse)), test_mse, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
plt.plot(np.linspace(0, len(train_mse) * cfg.batch_size, len(train_mse)), train_mse, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title('Epsilon Greedy')
plt.xlabel("Sample Size")
plt.ylabel("MSE")
plt.show()

top_y = top_y_rand
if len(top_y) < len(top_y_greedy):
    top_y = top_y_greedy

if len(top_y) < len(top_y_eps):
    top_y = top_y_eps

while len(top_mol_rand) < len(top_y):
    top_mol_rand.append(None)
while len(top_mol_greedy) < len(top_y):
    top_mol_greedy.append(None)
while len(top_mol_eps) < len(top_y):
    top_mol_eps.append(None)

plt.scatter(np.linspace(0, len(top_y) * cfg.batch_size, len(top_y)), top_mol_rand, label = 'Random')
plt.scatter(np.linspace(0, len(top_y) * cfg.batch_size, len(top_y)), top_mol_greedy, label = 'Greedy')
plt.scatter(np.linspace(0, len(top_y) * cfg.batch_size, len(top_y)), top_mol_eps, label = 'Epsilon Greedy')
plt.scatter(np.linspace(0, len(top_y) * cfg.batch_size, len(top_y)), top_y, label = 'Actual')
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.xlabel('Search Space')
plt.ylabel('Score')
plt.show()