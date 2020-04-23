#%% modules
import argparse
import random
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import DataLoader

from LambdaZero.chem import mol_to_graph
from LambdaZero.datasets.brutal_dock import BRUTAL_DOCK_DATA_DIR
from LambdaZero.environments.molecule import MPNNet

data_path = BRUTAL_DOCK_DATA_DIR.joinpath("d4/raw/dock_blocks105_walk40_clust.feather")
load_model_path = BRUTAL_DOCK_DATA_DIR.joinpath("d4/dock_blocks105_walk40_12_clust_model002")

desc = """ This script trains the Message Passing Neural Net used in LambdaZero
to predict the value of the reward """
usage = None
default_args = {
    'epochs': 10,
    'batch_size': 128,
    'lr': 0.0005,
    'maxgraphs': 100,
    'data': data_path,
    'loadmodel': load_model_path,
}

help_dict = {
    'data': "path to the `.parquet` dataset file",
    'maxgraphs': "maximum number of graphs built (set to 50 000 to use all)",
    'batch_size': "batch size used in the training dataloader",
}

print()
print("This script support args. Ask fro help:")
print("  python basic_mpnn_training_example.py --help")

### This is a loop to generate the `args` Namespace with argparse ##############
parser = argparse.ArgumentParser(description=desc, usage=usage)
for name, default in default_args.items():
    helpstr = ""
    try: 
        helpstr = help_dict[name]
    except (KeyError, NameError, TypeError): 
        pass
    if type(default) is list:
        parser.add_argument(f"--{name}", nargs="+", type=type(default[0]), default=default, help=helpstr)
    elif type(default) is bool:
        if default:
            parser.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"disables {name}")
        else:
            parser.add_argument(f"--{name}", action="store_true", default=default, help=helpstr)
    else:
        parser.add_argument("--" + name, type=type(default), default=default, help=helpstr)
args = parser.parse_known_args()[0]
################################################################################


#%% load dataframe
print()
print("loading data file")
dataframe = pd.read_feather(args.data)
smiles = dataframe['smiles'].values
scores = dataframe['gridscore'].to_numpy()
avg = scores.mean()
std = scores.std()
print("average dockscore:",avg)
print("standard deviation:", std)
normalized_scores = (scores - avg) / std


#%% build graphs (very long for the whole dataset)
# args.maxgraphs = 50000
print()
print(f"building up to {args.maxgraphs} graphs")
start = time.time()
dataset = [mol_to_graph(smiles[i], dockscore=normalized_scores[i]) for i in range(min(args.maxgraphs,len(smiles)))]
print(f"{len(dataset)} graphs built")
print(f"took {round(time.time()-start,3)} seconds")
print(f"occupy {sys.getsizeof(dataset)} bytes")

#standardize test set with seed 0
random.seed(0)
random.shuffle(dataset)
train_data = dataset[:int(0.8*len(dataset))]
valid_data = dataset[int(0.8*len(dataset)):int(0.9*len(dataset))]
test_data = dataset[int(0.9*len(dataset)):]

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)


#%% model
# copy of the implementation of LambdaZero.environment.molecule.MPNNet
class localMPNNet(torch.nn.Module):
    def __init__(self, num_feat=14, dim=64):
        super(localMPNNet, self).__init__()
        self.lin0 = nn.Linear(num_feat, dim)
        interaction_net = nn.Sequential(
            nn.Linear(4, 128), 
            nn.ReLU(), 
            nn.Linear(128, dim * dim)
        )
        self.conv = gnn.NNConv(dim, dim, interaction_net, aggr="mean")
        self.gru = nn.GRU(dim, dim)
        self.set2set = gnn.Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.MSELoss() 


#%% evaluate saved model
print()
print("evaluate saved model on those graphs")
saved_model = MPNNet().to(device)
saved_model.load_state_dict(torch.load(args.loadmodel, map_location=device))
saved_model.eval()
avg_loss = 0
for batch in valid_loader:
    batch = batch.to(device)
    pred = saved_model(batch)
    loss = loss_fn(pred, batch.dockscore)
    avg_loss += loss.item() * batch.num_graphs
print(f"  valid loss: {avg_loss/len(valid_data)}")


#%% training local model
print()
print("example of training loop on those graphs")
model = localMPNNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

trn_loss_history = []
vld_loss_history = []
for epoch in range(args.epochs):
    print(f"epoch {epoch+1}")
    
    model.train()
    avg_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.dockscore)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * batch.num_graphs
    print(f"  train loss: {avg_loss/len(train_data)}")
    trn_loss_history.append(avg_loss/len(train_data))

    model.eval()
    avg_loss = 0
    for batch in valid_loader:
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.dockscore)
            avg_loss += loss.item() * batch.num_graphs
    print(f"  valid loss: {avg_loss/len(valid_data)}")
    vld_loss_history.append(avg_loss/len(valid_data))




#%%
plt.plot(trn_loss_history, label="train")
plt.plot(vld_loss_history, label="valid")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()