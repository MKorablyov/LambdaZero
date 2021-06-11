# -*- coding: utf-8 -*-
"""
Training a Classifier
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn

from LambdaZero.environments import BlockMolEnvGraph_v1
from LambdaZero.models.torch_graph_models import fast_from_data_list


class CustomDataset(Dataset):
    def __init__(self, obss, targets):
        self._obss = obss
        self._targets = targets

    def __getitem__(self, idx):
        sample = {'obs': self._obss[idx], 'target': self._targets[idx]}
        return sample

    def __len__(self):
        return len(self._obss)


def collate_fn(data):
    obss = dict()
    for k in data[0]["obs"].keys():
        obss[k] = np.stack([x["obs"][k] for x in data])
    _tgt = torch.stack([x["target"] for x in data])
    return obss, _tgt


# ==================================================================================================
# Train config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 68
train_split = 1.0

# ==================================================================================================
# Load dataset

# Load env_data
env_data = np.load("data/full_env_data.npy", allow_pickle=True).item()

# Get unpacking method
env = BlockMolEnvGraph_v1({"random_steps": 3, "allow_removal": True, "max_steps": 1})
env_space = env.observation_space["mol_graph"]
_unpack = env_space.unpack


# Construct data
_inputs, _targets = [], []
for smi, smi_data in env_data.items():
    x = smi_data["obs"]
    act_mask = smi_data["obs"]["action_mask"].astype(np.bool)
    act_r_df = smi_data["next"]["true_r"]
    tgt = torch.zeros(act_mask.shape)
    # tgt[~act_mask] = -20  # as per logits set
    tgt[act_r_df.index.values] = torch.from_numpy(act_r_df.values).float()

    _inputs.append(x)
    _targets.append(tgt)

# Process Data
# _inputs = fast_from_data_list(_inputs)
_targets = torch.stack(_targets)
_targets = torch.argmax(_targets, dim=1)
_targets = _targets.to(device)

# Create dataloaders
rand_idxs = torch.randperm(len(_inputs))
train_idx = rand_idxs[:round(len(_inputs) * train_split)]
test_idx = rand_idxs[len(train_idx):]


def get_idxs(idxs):
    _x = [_inputs[i] for i in idxs]
    _y = _targets[idxs]
    return _x, _y


train_set = CustomDataset(*get_idxs(train_idx))
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_set = CustomDataset(*get_idxs(test_idx))
val_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

tgtmatch = dict()
molobs = []
for x, y in zip(_inputs, _targets):
    molobs.append(x["mol_graph"])
    tgtmatch[tuple(x["mol_graph"])] = y
molobs = np.stack(molobs)

obs_non_zero_idx = np.where(molobs.sum(axis=0) != 0)

for i_batch, sample_batched in enumerate(train_loader):
    obs, tgt = sample_batched
    for ooo, lll in zip(obs["mol_graph"], tgt):
        if not (lll == tgtmatch[tuple(ooo)]):
            print("WTFFF")

    print(i_batch, obs["mol_graph"].shape, tgt.size())

# ==================================================================================================


def train_epoch(loader, model, optimizer, criterion):
    model.train()

    correct = 0
    total = 0

    running_loss = 0.0
    for bidx, (data_obs, data_tgt) in enumerate(loader):
        optimizer.zero_grad()
        y_hat = model(data_obs)
        loss = criterion(y_hat, data_tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(y_hat.data, 1)
        total += data_tgt.size(0)
        correct += (predicted == data_tgt).sum().item()

    return correct / total, running_loss

# ==================================================================================================
# Load net


from LambdaZero.models.model_atom import MPNNet_v2
from LambdaZero.models.model_egnn import EGNNetRLBO


class GraphNet(nn.Module):
    def __init__(self, model_class, env, device, dim=256):
        super().__init__()

        model = model_class(num_feat=env_space.num_node_feat,
                            dim=dim, num_out_per_stem=env.num_blocks)
        model = model.to(device)
        self.space = env.observation_space["mol_graph"]
        self.model = model
        self.device = device

    def forward(self, obs):
        device = self.device

        # action_mask = obs["action_mask"]
        enc_graphs = obs["mol_graph"].astype(np.uint8)
        graphs = [self.space.unpack(i) for i in enc_graphs]
        data = fast_from_data_list(graphs)
        data = data.to(device)
        # </end of expensive unpacking> The rest of this is the
        # forward pass (~5ms) and then a backward pass + update (which
        # I can't measure directly but seems to take at most ~20ms)

        scalar_outs, data = self.model(data)
        stop_logit = scalar_outs[:, 1:2]
        break_logits = data.jbond_preds.reshape((data.num_graphs, -1))
        add_logits = data.stem_preds.reshape((data.num_graphs, -1))

        actor_logits = torch.cat([stop_logit,
                                  break_logits,
                                  add_logits], 1)

        return actor_logits


class GraphNetMLP(nn.Module):
    def __init__(self, model_class, env, device, dim=512):
        super().__init__()
        in_size = env.observation_space['mol_graph'].shape[0]

        self.model = nn.Sequential(
            nn.Linear(in_size, dim), nn.LeakyReLU(),
            nn.Linear(dim, dim), nn.LeakyReLU(),
            nn.Linear(dim, dim), nn.LeakyReLU(),
        )
        self._policy = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(),
            nn.Linear(dim, env.action_space.n),
        )

        self.space = env.observation_space["mol_graph"]
        self.device = device

    def forward(self, obs):
        device = self.device

        # action_mask = obs["action_mask"]
        data = torch.from_numpy(obs["mol_graph"])
        data = data.to(device).float()
        data.div_(255)

        feat = self.model(data)
        actor_logits = self._policy(feat)

        return actor_logits


model = GraphNet(MPNNet_v2, env, device, dim=128)
# model = GraphNetMLP(MPNNet_v2, env, device)
model = model.to(device)

# print(model(obs))
# ==================================================================================================
# Train

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(1000):  # loop over the dataset multiple times
    acc, loss = train_epoch(train_loader, model, optimizer, criterion)
    print(acc, loss)
