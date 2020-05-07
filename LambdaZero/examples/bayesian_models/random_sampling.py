import pandas as pd
import numpy as np
import torch as th
import random
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns

class cfg:
    # data could be found here: https://github.com/MKorablyov/brutal_dock/tree/master/d4/raw
    data = 'd4_10k_clust.parquet'#"/home/maksym/Datasets/brutal_dock/d4/raw/d4_100k_clust.parquet"
    batch_size = 500
    epochs = 500

#sns.distplot(df["dockscore"])
#plt.show()

# total_budget = 3000

# strategies
# S1: greedy
# batch_size = 500, num_interations = 6

class Model(th.nn.Module):
    def __init__(self, inp_size, output_size):
        super(Model, self).__init__()
        self.fc1 = th.nn.Linear(1024, 512)
        self.relu = th.nn.ReLU()
        self.fc2 = th.nn.Linear(512, output_size)

    def forward(self, x):
        fc = self.fc1(x)
        rel = self.relu(fc)
        #out1 = self.relu(self.fc1(x))
        return self.fc2(rel)

def train(model, states, labels):
    batch_size = cfg.batch_size
    net  = model
    
    epochs = cfg.epochs
    criterion =  th.nn.MSELoss()
    optimizer = th.optim.SGD(net.parameters(), lr = 0.001)
    running_loss = 0

    for _ in range(epochs):
        for i, j in enumerate(states):
            x = j
            #print(labels[i])
            y = labels[i] #th.from_numpy(labels[i].astype(np.float32))
            #print(x)
            optimizer.zero_grad()
            #print(x.shape)
            outputs = net(x)
            loss = criterion(outputs, th.tensor([y], dtype = th.float32))
            loss.backward()
            optimizer.step()
            running_loss += loss

    return model, running_loss

def evaluate(model, states, labels):
    criterion = th.nn.MSELoss()
    predictions = model(states)
    MSE = criterion(predictions, th.tensor([labels], dtype = th.float32))

    return predictions, MSE

if __name__ == '__main__':
    df = pd.read_parquet(cfg.data)
    #print(df['fingerprint'].values[0].shape)
    
    x = th.from_numpy(np.vstack(df['fingerprint'].values).astype(np.float32))
    y = list(df['dockscore'].values)#th.from_numpy(np.vstack(df['dockscore'].values).astype(np.float32))
    model = Model(x.shape[0], 1)
    model, MSE = train(model, x, y)
    print('running MSE after training: {}'.format(MSE.item()))
    predictions, MSE = evaluate(model, x, y)
    print('MSE after eval: {}'.format(MSE.item()))
