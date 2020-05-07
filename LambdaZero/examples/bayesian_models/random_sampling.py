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
    data = 'd4_250k_clust.parquet'#"/home/maksym/Datasets/brutal_dock/d4/raw/d4_100k_clust.parquet"
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
    
    loss_list = []
    epochs = cfg.epochs
    criterion =  th.nn.MSELoss()
    criterion2 =  th.nn.L1Loss()
    optimizer = th.optim.SGD(net.parameters(), lr = 0.001)

    for _ in range(epochs):
        for j in range(0, len(states), batch_size):
            x = states[j:j+batch_size]
            y = labels[j:j + batch_size] 
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, th.tensor([y], dtype = th.float32))
            loss2 = criterion2(outputs, th.tensor([y], dtype = th.float32))
            loss.backward()
            optimizer.step()     

    return model, loss, loss2

def evaluate(model, states, labels):
    criterion = th.nn.MSELoss()
    criterion2 = th.nn.L1Loss()

    predictions = model(states)
    MSE = criterion(predictions, th.tensor([labels], dtype = th.float32))
    MAE = criterion2(predictions, th.tensor([labels], dtype = th.float32))

    return predictions, MSE, MAE

if __name__ == '__main__':
    sample_size = [10,100, 1000, 10000,100000]#9558
    test_mae = []
    train_mae = []
    df1 = pd.read_parquet(cfg.data)
    for i in sample_size:
        df = df1.sample(n = i)
        x = th.from_numpy(np.vstack(df['fingerprint'].values).astype(np.float32))
        y = list(df['dockscore'].values)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        
        model = Model(x.shape[0], 1)
        model, MSE, MAE = train(model, x_train, y_train)
        train_mae.append(MAE)
        print('running MAE after training: {}'.format(MAE.item()))
        predictions, MSE, MAE = evaluate(model, x_test, y_test)
        test_mae.append(MAE)
        print('MAE after eval: {}'.format(MAE.item()))

    plt.plot(sample_size, test_mae, color="r", linestyle="-", marker="^", linewidth=1,label="Test")
    plt.plot(sample_size, train_mae, color="b", linestyle="-", marker="s", linewidth=1,label="Train")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Sample Size")
    plt.ylabel("MAE")
    plt.show()