import pandas as pd
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F
# import seaborn as sns

device = 'cuda:0' if th.cuda.is_available() else 'cpu'

class cfg:
    # data could be found here: https://github.com/MKorablyov/brutal_dock/tree/master/d4/raw
    data = 'd4_250k_clust.parquet'#"/home/maksym/Datasets/brutal_dock/d4/raw/d4_100k_clust.parquet"
    batch_size = 500
    epochs = 6

#sns.distplot(df["dockscore"])
#plt.show()

# total_budget = 3000

# strategies
# S1: greedy
# batch_size = 500, num_interations = 6

# Notes: I didn't use the random_split function b/c it needs data_loader function.
def random_split(dataset, test_prob, valid_prob,
                  test_idx=th.tensor([], dtype=th.long),
                  train_idx=th.tensor([], dtype=th.long),
                  val_idx=th.tensor([], dtype=th.long)):
    # todo assert order
    num_last_split = (len(test_idx) + len(train_idx) + len(val_idx))
    num_split = len(dataset) - num_last_split

    ntest = int(num_split * test_prob)
    nvalid = int(num_split * valid_prob)
    idx = th.randperm(num_split) + num_last_split

    test_idx = th.cat([test_idx, idx[:ntest]])
    val_idx = th.cat([val_idx, idx[ntest:ntest + nvalid]])
    train_idx = th.cat([train_idx, idx[ntest + nvalid:]])

    return test_idx, val_idx, train_idx

def train_epoch(x, y, model, optimizer):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    for i in tqdm(range(0, len(x), batch_size)):
        x_in = x[i:i+batch_size]
        x_in = x_in.to(device)
        y_in = y[i:i+batch_size]
        y_in = y_in.to(device)
        optimizer.zero_grad()
        preds = model(x_in)
        loss = F.mse_loss(preds.squeeze(1), y_in)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
        losses.append(loss)
    return loss_all / len(x), loss

def train_greedy(x, y, model, optimizer):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    for i in tqdm(range(0, len(x), batch_size)):
        x_in = x[i:i+batch_size]
        x_in = x_in.to(device)
        y_in = y[i:i+batch_size]
        y_in = y_in.to(device)
        optimizer.zero_grad()
        preds = model(x_in)
        loss = F.mse_loss(preds.squeeze(1), y_in)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
        losses.append(loss)
    return loss_all / len(x), loss

def train_greedy_error(x, y, model, optimizer):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    bound = 0

    for i in tqdm(range(0, len(x), batch_size)):
        if bound == 0:
            bound = i + batch_size
        x_in = x[0:bound]
        x_in = x_in.to(device)
        y_in = y[0:bound]
        y_in = y_in.to(device)

        x_rest = x[bound:]
        x_rest = x_in.to(device)
        y_rest = y[bound:]
        y_rest = y_in.to(device)

        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            preds = model(x_in)
            loss = F.mse_loss(preds.squeeze(1), y_in)
            loss.backward()
            optimizer.step()
        preds = model(x_rest)
        print(preds.shape)
        
        tmp = np.concatenate((x_rest.detach().cpu().numpy(), preds.detach().cpu().numpy()), axis = 2)
        tmp = np.sort(tmp, axis = 2)
        print(tmp.shape)
        bests = preds[:i * batch_size]
        loss_all += loss.item()
        losses.append(loss)
    return loss_all / len(x), loss

def test_epoch(x, y, model):
    model.eval()
    ys = []
    preds = []
    batch_size = cfg.batch_size
    for i in tqdm(range(0, len(x), batch_size)):
        x_in = x[i:i+batch_size]
        x_in = x_in.to(device)
        y_in = y[i:i+batch_size]
        y_in = y_in.to(device)
        ys.append(y_in.detach().cpu().numpy())
        preds.append(model(x_in).detach().cpu().numpy())

    ys = np.concatenate(ys, axis=0)
    preds = np.concatenate(preds, axis=0)
    mse = np.mean((ys - preds) ** 2)
    mae = np.mean(np.abs(ys - preds))
    #test_out = pd.DataFrame({"ys": ys, "preds": preds})
    return mse, mae, ys, preds


class Model(th.nn.Module):
    def __init__(self, inp_size, output_size):
        super(Model, self).__init__()
        self.fc1 = th.nn.Linear(1024, 512)
        self.relu = th.nn.ReLU()
        self.fc2 = th.nn.Linear(512, output_size)

    def forward(self, x):
        fc = self.fc1(x)
        rel = self.relu(fc)
        return self.fc2(rel)

if __name__ == '__main__':
    sample_size = [10,100, 1000, 10000,100000]#9558

    test_mae = []
    train_mae = []
    df1 = pd.read_parquet(cfg.data)
    #df = df1.sample()
    x = th.from_numpy(np.vstack(df1['fingerprint'].values).astype(np.float32))
    y = th.Tensor(list(df1['dockscore'].values))
    model = Model(1024, 1)#(x.shape[0], 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

    optimizer = th.optim.Adam(model.parameters())

    train_errors = []
    test_errors = []
    val_errors = []

    for i in range(cfg.epochs):
        
        tmp = list(zip(x_train, y_train))
        random.shuffle(list(tmp))
        x_train, y_train = zip(*tmp)
        x_train = np.asarray([t.numpy() for t in x_train], dtype = np.float32)
        y_train = np.asarray([t.numpy() for t in y_train], dtype = np.float32)
        x_train = th.FloatTensor(x_train)
        y_train = th.FloatTensor(y_train)

        avg_loss, train_mse = train_epoch(x_train, y_train, model, optimizer)
        mse, mae, ys, preds = test_epoch(x_test, y_test, model)

        train_errors.append(train_mse)
        test_errors.append(mse)
    
        mse, mae, ys, preds = test_epoch(x_val, y_val, model)
        val_errors.append(mse)

    plt.plot(np.linspace(0, len(x_test), 6), test_errors, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()
    plt.plot(np.linspace(0, len(x_train), 6), train_errors, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()
    plt.plot(np.linspace(0, len(x_val), 6), val_errors, color="g", linestyle="-", marker="c", linewidth=1,label="Val")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()