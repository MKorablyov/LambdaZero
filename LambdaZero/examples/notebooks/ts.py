import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

class cfg:
    num_feat = 1
    dim = 128
    train_prob = 0.05

class Model(nn.Module):
    def __init__(self, num_feat=cfg.num_feat, ksizes=[12, 8, 6], dim=cfg.dim, act=torch.nn.Tanh, drop_prob=0.1):
        nn.Module.__init__(self)
        self.drop_prob=drop_prob
        self.l0 = nn.Sequential(nn.Conv1d(num_feat, dim, ksizes[0], padding=ksizes[0]//2),act(), nn.MaxPool1d(2))
        self.l1 = nn.Sequential(nn.Conv1d(dim, dim, ksizes[1], padding=ksizes[1]//2), act(), nn.MaxPool1d(2))
        self.l2 = nn.Sequential(nn.Conv1d(dim, dim, ksizes[2], padding=ksizes[2]//2), act(), nn.MaxPool1d(2))
        self.l3 = nn.Linear(dim//8 * dim,1)

    def forward(self, x, do_dropout):
        h0 = F.dropout(self.l0(x), training=do_dropout, p=self.drop_prob)
        h1 = F.dropout(self.l1(h0), training=do_dropout, p=self.drop_prob)
        h2 = F.dropout(self.l2(h1), training=do_dropout, p=self.drop_prob)
        h2_flat = torch.flatten(h2,start_dim=1)
        h3 = self.l3(h2_flat)
        return h3

# make daaset
x_ = np.linspace(0,1, num=500)
y_ = np.sin(x_ * 16*np.pi - 8*np.pi) * np.abs(x_ - 0.5)
x = torch.tensor(x_*(cfg.dim-1), dtype=torch.int64)
x = torch.nn.functional.one_hot(x,num_classes=cfg.dim)[:,None,:].float()
y = torch.tensor(y_)



idx = np.arange(x.shape[0])
np.random.shuffle(idx)
num_train = int(cfg.train_prob * x.shape[0])
train_idx, val_idx = idx[:num_train], idx[num_train:]

train_idx = train_idx[np.logical_and((x_[train_idx] > 0.15), (x_[train_idx] < 0.85))]


train_ord, val_ord = np.argsort(train_idx), np.argsort(val_idx)
train_idx,val_idx = train_idx[train_ord], val_idx[val_ord]
x_train, y_train, x_val, y_val = x[train_idx], y[train_idx], x[val_idx], y[val_idx]
x_train_, y_train_, x_val_, y_val_ = x_[train_idx], y_[train_idx], x_[val_idx], y_[val_idx]


def _train_model():
    model = Model()
    # train_model
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(40):
        optim.zero_grad()
        preds_train = model(x_train, do_dropout=True)[:, 0]
        loss_train = ((preds_train - y_train) ** 2).mean()
        loss_train.backward()
        optim.step()
        preds_val = model(x_val, do_dropout=False)[:, 0]
        loss_val = ((preds_val - y_val) ** 2).mean()

        print("train_loss", "%.5f" % loss_train.detach().cpu().numpy(),
              "val_loss", "%.5f" % loss_val.detach().cpu().numpy())
    return model

#model = _train_model()

# do Thompson sampling
samples_val = []
for i in range(10):
    model = _train_model()
    sample_val = model(x_val, do_dropout=True)[:, 0].detach().cpu().numpy()
    samples_val.append(sample_val)
samples_val = np.stack(samples_val,axis=1)
sele_idx = np.argmax(samples_val, axis=0)

print(samples_val.max())

print(sele_idx.shape)

sele_x = x_val_[sele_idx]
sele_preds = [(samples_val[:,i])[int(sele_idx[i])] for i in range(samples_val.shape[1])]

plt.plot(x_,y_)
plt.scatter(x_train_, y_train_, marker="o", s=250)
[plt.scatter(x_val_, s) for s in samples_val.T]
plt.scatter(sele_x,sele_preds,marker="X",s=250)

plt.show()