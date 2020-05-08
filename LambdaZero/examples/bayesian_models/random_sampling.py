import pandas as pd
import numpy as np
import torch as th
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

#sns.distplot(df["dockscore"])
#plt.show()

# total_budget = 3000

# strategies
# S1: greedy
# batch_size = 500, num_interations = 6

def train_epoch(x, y, model, optimizer, x_test, y_test):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    test_mse = []
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
    
        mse, _, _, _ = test_epoch(x_test, y_test, model)
        test_mse.append(mse)
    return loss_all / len(x), losses, test_mse

def train_greedy(x, y, model, optimizer, x_test, y_test):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    test_mse = []
    bound = 0
    model = model.to(device)

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

        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            preds = model(x_in)
            loss = F.mse_loss(preds.squeeze(1), y_in)
            loss.backward()
            optimizer.step()
        
        loss_all += loss.item()
        losses.append(loss)

        mse, _, _, _ = test_epoch(x_test, y_test, model)
        test_mse.append(mse)

        try:
            preds = model(x_rest)
            
            ind = preds.argsort()
            sorted_y = y_rest[ind].detach().cpu().numpy().squeeze()
            sorted_x = x_rest[ind].detach().cpu().numpy().squeeze()

            x_in = np.concatenate((x_in.detach().cpu().numpy(), sorted_x[-cfg.batch_size:]))
            y_in = np.concatenate((y_in.detach().cpu().numpy(), sorted_y[-cfg.batch_size:]))
            x_rest = sorted_x[:-cfg.batch_size]
            y_rest = sorted_y[:-cfg.batch_size]

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

    return loss_all / len(x), losses, test_mse

def train_greedy_error(x, y, model, optimizer, x_test, y_test):
    model.train()
    batch_size = cfg.batch_size
    loss_all = 0
    losses = []
    test_mse = []
    bound = 0
    std = cfg.std
    mean = cfg.mean
    model = model.to(device)

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


        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            preds = model(x_in)
            loss = F.mse_loss(preds.squeeze(1), y_in)
            loss.backward()
            optimizer.step()
        
        loss_all += loss.item()
        losses.append(loss)

        mse, _, _, _ = test_epoch(x_test, y_test, model)
        test_mse.append(mse)

        try:
            preds = model(x_rest)
            dist = th.randn(preds.size()) * std + mean
            dist = dist.to(device)
            preds = preds + dist
            
            ind = preds.argsort()
            sorted_y = y_rest[ind].detach().cpu().numpy().squeeze()
            sorted_x = x_rest[ind].detach().cpu().numpy().squeeze()

            x_in = np.concatenate((x_in.detach().cpu().numpy(), sorted_x[-cfg.batch_size:]))
            y_in = np.concatenate((y_in.detach().cpu().numpy(), sorted_y[-cfg.batch_size:]))
            x_rest = sorted_x[:-cfg.batch_size]
            y_rest = sorted_y[:-cfg.batch_size]

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

        
    return loss_all / len(x), losses, test_mse

def test_epoch(x, y, model):
    model.eval()
    ys = []
    preds = []
    batch_size = cfg.batch_size
    for i in range(0, len(x), batch_size):
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
    df1 = pd.read_parquet(cfg.data)
    x = th.from_numpy(np.vstack(df1['fingerprint'].values).astype(np.float32))
    y = th.Tensor(list(df1['dockscore'].values))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

    #EPSILON GREEDY
    model = Model(1024, 1)#(x.shape[0], 1)
    model = model.to(device)

    optimizer = th.optim.Adam(model.parameters())
        
    tmp = list(zip(x_train, y_train))
    random.shuffle(list(tmp))
    x_train, y_train = zip(*tmp)
    x_train = np.asarray([t.numpy() for t in x_train], dtype = np.float32)
    y_train = np.asarray([t.numpy() for t in y_train], dtype = np.float32)
    x_train = th.FloatTensor(x_train)
    y_train = th.FloatTensor(y_train)

    avg_loss, train_mse, test_mse = train_greedy_error(x_train, y_train, model, optimizer, x_test, y_test)
    mse, mae, ys, preds = test_epoch(x_val, y_val, model)

    plt.plot(np.linspace(0, len(test_mse) * cfg.batch_size, len(test_mse)), test_mse, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
    plt.plot(np.linspace(0, len(train_mse) * cfg.batch_size, len(train_mse)), train_mse, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title('Epsilon Greedy')
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()

    #GREEDY
    model = Model(1024, 1)#(x.shape[0], 1)
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters())
        
    tmp = list(zip(x_train, y_train))
    random.shuffle(list(tmp))
    x_train, y_train = zip(*tmp)
    x_train = np.asarray([t.numpy() for t in x_train], dtype = np.float32)
    y_train = np.asarray([t.numpy() for t in y_train], dtype = np.float32)
    x_train = th.FloatTensor(x_train)
    y_train = th.FloatTensor(y_train)

    avg_loss, train_mse, test_mse = train_greedy(x_train, y_train, model, optimizer, x_test, y_test)
    mse, mae, ys, preds = test_epoch(x_val, y_val, model)


    plt.plot(np.linspace(0, len(test_mse) * cfg.batch_size, len(test_mse)), test_mse, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
    plt.plot(np.linspace(0, len(train_mse) * cfg.batch_size, len(train_mse)), train_mse, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title('Greedy')
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()

    #STANDARD
    model = Model(1024, 1)#(x.shape[0], 1)
    model = model.to(device)

    optimizer = th.optim.Adam(model.parameters())
        
    tmp = list(zip(x_train, y_train))
    random.shuffle(list(tmp))
    x_train, y_train = zip(*tmp)
    x_train = np.asarray([t.numpy() for t in x_train], dtype = np.float32)
    y_train = np.asarray([t.numpy() for t in y_train], dtype = np.float32)
    x_train = th.FloatTensor(x_train)
    y_train = th.FloatTensor(y_train)

    avg_loss, train_mse, test_mse = train_epoch(x_train, y_train, model, optimizer, x_test, y_test)
    mse, mae, ys, preds = test_epoch(x_val, y_val, model)

    plt.plot(np.linspace(0, len(test_mse) * cfg.batch_size, len(test_mse)), test_mse, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
    plt.plot(np.linspace(0, len(train_mse) * cfg.batch_size, len(train_mse)), train_mse, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title('Standard')
    plt.xlabel("Sample Size")
    plt.ylabel("MSE")
    plt.show()