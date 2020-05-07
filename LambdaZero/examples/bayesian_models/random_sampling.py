import pandas as pd
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns

class cfg:
    # data could be found here: https://github.com/MKorablyov/brutal_dock/tree/master/d4/raw
    data = 'd4_10k_clust.parquet'#"/home/maksym/Datasets/brutal_dock/d4/raw/d4_100k_clust.parquet"
    batch_size = 500
    epochs = 6

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
        return self.fc2(rel)

def train(model, states, labels):
    batch_size = cfg.batch_size
    net  = model
    
    epochs = cfg.epochs
    criterion =  th.nn.MSELoss()
    optimizer = th.optim.SGD(net.parameters(), lr = 0.001)

    for _ in range(epochs):
        for j in range(0, len(states), batch_size):
            x = states[j:j+batch_size]
            y = labels[j:j + batch_size] 
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, th.tensor([y], dtype = th.float32))
            loss.backward()
            optimizer.step()
            

    return model, loss

def evaluate(model, states, labels):
    criterion = th.nn.MSELoss()
    predictions = model(states)
    MSE = criterion(predictions, th.tensor([labels], dtype = th.float32))

    return predictions, MSE

if __name__ == '__main__':
    df = pd.read_parquet(cfg.data)
    x = th.from_numpy(np.vstack(df['fingerprint'].values).astype(np.float32))
    y = list(df['dockscore'].values)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    
    model = Model(x.shape[0], 1)
    model, MSE = train(model, x_train, y_train)
    print('running MSE after training: {}'.format(MSE.item()))
    predictions, MSE = evaluate(model, x_test, y_test)
    print('MSE after eval: {}'.format(MSE.item()))
