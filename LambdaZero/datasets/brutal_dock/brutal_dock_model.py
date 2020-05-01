import argparse
import json
import os
import socket
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops

from LambdaZero import inputs

# import seaborn as sns
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args(args):

    parser = argparse.ArgumentParser(description='Train a reward model (brutal dock)')

    parser.add_argument('--log', type=str, default=None,
                        hlep='If specified, path to the file used as log. Defaults to None (log to terminal)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='path to dataset')

    parser.add_argument('--debug', default=False, action='store_true',
                        help='If specified, train on only a small dataset')

    def_ckpt = 'saved_model/'
    parser.add_argument('--ckpt_path', type=str, default=def_ckpt,
                        help=f'Checkpointing path. Defaults to {def_ckpt}')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the json file containing the configuration of the model.')

    return parser.parse_args(args)


class cfg:
    machine = socket.gethostname()
    if machine == "Ikarus":
        datasets_dir = "/home/maksym/Datasets"
        programs_dir = "/home/maksym/Programs"
        summaries_dir = "/home/maksym/Desktop/model_summaries"
    else:
        datasets_dir = "/home/mkkr/scratch/Datasets"
        programs_dir = "/home/mkkr/Programs"
        summaries_dir = "/home/mkkr/scratch/model_summaries"

    load_model = None #os.path.join(datasets_dir, "brutal_dock/d4/d4_100k_mine_model_001")
    db_root = os.path.join(datasets_dir, "brutal_dock/d4")
    db_name = "dock_blocks105_walk40_12_clust"
    molprops = ["gridscore", "klabel"]
    target = "gridscore"
    file_names = ["dock_blocks105_walk40_clust", "dock_blocks105_walk40_2_clust"] # fixme ifcoord in molmdp
    target_norm = [-26.3, 12.3]
    test_prob = 0.1
    valid_prob = 0.1
    b_size = 16
    dim = 64
    num_epochs = 120
    outpath = "/home/maksym/Desktop/model_summaries/brutal_dock"
    model_name = db_name + "model002"

#df = pd.read_feather(os.path.join(cfg.db_root, "raw", cfg.file_names[1])+".feather")
#gridscores = df["gridscore"].to_numpy()
#print(gridscores.mean(), gridscores.std())
#time.sleep(200)


class MyTransform(object):
    def __init__(self,target_norm,target=cfg.target):
        self.target = target
        self.target_norm = target_norm
    def __call__(self, data):
        # Specify target.
        y = getattr(data, self.target)
        y = (y - self.target_norm[0]) / self.target_norm[1]
        data.y = y
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device
        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        return data


# to do: move to a different .py file
class MPNN(torch.nn.Module):
    def __init__(self,
                 input_feat: int =14,
                 gcn_in: int = 10,
                 gcn_out: int = 128,
                 nedge: int = 4,
                 edge_hidden: int = 128,
                 gru_out: int = 128,
                 gru_layers: int = 1,
                 linear_hidden: int = 128,
                 out_size: int = 1
                 ):
        """
        message passing neural network.

        Args:
            input_feat (int, optional): number of input features. Defaults to 14.
            gcn_in (int, optional): size of GCN inputs size. Defaults to 10.
            gcn_out (int, optional): size of GCN outsize embedding size. Defaults to 128.
            nedge (int, optional): number of edge features. Defaults to 4.
            edge_hidden (int, optional): edge hidden embedding size. Defaults to 128.
            gru_out (int, optional): size out GRU output. Defaults to 128.
            gru_layers (int, optional): number of layers in GRU. Defaults to 1.
            linear_hidden (int, optional): hidden size in fully-connected network. Defaults to 128.
            out_size (int, optional): output size. Defaults to 1.
        """
        super(MPNN, self).__init__()
        self.lin0 = nn.Linear(input_feat, gcn_in)

        edge_network = nn.Sequential(
            nn.Linear(nedge, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, gcn_in * gcn_out)
        )

        self.conv = NNConv(gcn_in, gcn_out, edge_network, aggr='mean')
        self.gru = nn.GRU(gcn_out, gru_out, num_layers=gru_layers)

        self.set2set = Set2Set(gru_out, processing_steps=3)
        self.fully_connected = nn.Sequential(
            nn.Linear(2 * gru_out, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, out_size)
        )

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = self.fully_connected(out)
        return out


def _random_split(dataset, test_prob, valid_prob,
                  test_idx=torch.tensor([], dtype=torch.long),
                  train_idx=torch.tensor([], dtype=torch.long),
                  val_idx=torch.tensor([], dtype=torch.long)):
    # todo assert order
    num_last_split = (len(test_idx) + len(train_idx) + len(val_idx))
    num_split = len(dataset) - num_last_split

    ntest = int(num_split * test_prob)
    nvalid = int(num_split * valid_prob)
    idx = torch.randperm(num_split) + num_last_split

    test_idx = torch.cat([test_idx, idx[:ntest]])
    val_idx = torch.cat([val_idx, idx[ntest:ntest + nvalid]])
    train_idx = torch.cat([train_idx, idx[ntest + nvalid:]])

    return test_idx, val_idx, train_idx


def knn_split(klabels, probs):
    "split the dataset based on the KNN cluster assignment in some space"
    klabels = np.asarray(klabels)
    nsplits = len(probs)
    assert len(klabels.shape) == 1, "bad input shape for split"
    assert np.isclose(np.sum(probs), 1.0), "probabilities do not sum up to 1"

    klabel_sort = np.argsort(klabels)
    klabel_unq, klabel_count = np.unique(klabels, return_counts=True)
    klabel_slices = np.concatenate([np.asarray([0]), np.cumsum(klabel_count)],0)

    splits = [[] for _ in range(nsplits)]
    for i, klabel in enumerate(klabel_unq):
        klabel_idx = klabel_sort[klabel_slices[i]:klabel_slices[i + 1]]
        draw = np.random.choice(np.arange(nsplits), 1, p=probs, replace=True)[0]
        splits[draw].append(klabel_idx)
    splits = (np.concatenate(sp,0) for sp in splits)
    return splits


class Environment:
    def __init__(self, target_norm, test_prob, valid_prob, b_size, outpath, config, load_model=None):
        """
        environment variables. Will be modified in next commits.
        Args doc to be reviewed.

        Args:
            target_norm: list. target normalization values. TBC
            test_prob:
            valid_prob:
            b_size:
            outpath:
            config: dict. parsed json file - arguments for model
            load_model (optional): ... Defaults to None.
        """
        self.target_norm = target_norm
        self.test_prob = test_prob
        self.valid_prob = valid_prob
        self.b_size = b_size
        # load and normalize dataset
        self.transform = T.Compose([MyTransform(target_norm), Complete(),]) #  T.Distance(norm=False)

        # make model
        model_name = config['model']['name']
        if model_name != 'MPNN':
            raise ValueError(f'Model type {model_name} is not implemented.')
        # initialize model
        self.model = MPNN(config['model']).to(device)
        if load_model is not None: self.model.load_state_dict(torch.load(load_model))
        if not os.path.exists(outpath): os.makedirs(outpath)
        self.outpath = outpath

    def load_dataset(self, db_root, file_names):

        dataset = inputs.BrutalDock(db_root, props=cfg.molprops, transform=self.transform, file_names=file_names)
        self.dataset = dataset

        # Split datasets.
        klabels = [dataset[i].klabel for i in range(len(dataset))]
        test_probs = [self.test_prob, self.valid_prob, 1. - self.test_prob - self.valid_prob]
        splits = knn_split(klabels,test_probs)
        self.test_idx, self.val_idx, self.train_idx = (torch.from_numpy(sp) for sp in splits)

        # if reset_split:
        #if reset_split:
        #    self.test_idx, self.val_idx, self.train_idx = _group_split(, self.test_prob, self.valid_prob)
        # else:
        #     self.test_idx, self.val_idx, self.train_idx = _knn_split(dataset, self.test_prob, self.valid_prob,
        #                                                                 test_idx=self.test_idx,
        #                                                                 val_idx=self.val_idx,
        #                                                                 train_idx=self.train_idx)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=self.b_size, shuffle=False)
        self.val_loader = DataLoader(self.dataset[self.val_idx], batch_size=self.b_size, shuffle=False)
        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=self.b_size, shuffle=True)

    def train_epoch(self, loader, model, optimizer):
        model.train()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            preds = model(data)
            loss = F.mse_loss(preds, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_graphs
        return loss_all / len(loader.dataset)

    def test_epoch(self, loader, model):
        model.eval()
        ys = []
        preds = []
        for data in loader:
            data = data.to(device)
            ys.append(data.y.detach().cpu().numpy())
            preds.append(model(data).detach().cpu().numpy())

        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=0)
        mse = np.mean((ys - preds) ** 2)
        mae = np.mean(np.abs(ys - preds))
        test_out = pd.DataFrame({"ys": ys, "preds": preds})
        return mse, mae, test_out

    def train_model(self, num_epochs, save_model, lr=0.001, sched_factor=0.7, sched_patience=5, min_lr=0.00001,
                    load_model=None):

        if load_model is not None: self.model.load_state_dict(torch.load(load_model))
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sched_factor,
                                                            patience=sched_patience, min_lr=min_lr)
        best_val_error = None
        for epoch in range(1, num_epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_error = self.train_epoch(self.train_loader, self.model, optimizer)
            val_error, val_mae, val_df = self.test_epoch(self.val_loader, self.model)
            scheduler.step(val_error)
            if best_val_error is None or val_error <= best_val_error:
                test_error, test_mae, test_df = self.test_epoch(self.test_loader, self.model)
                best_val_error = val_error
                torch.save(self.model.state_dict(), os.path.join(self.outpath, save_model))  # save best model
            print('Epoch: {:03d}, LR: {:5f}, Train Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f} '
                  .format(epoch, lr, train_error, val_error, test_error), end="")
            print("Test MAE {:.5f} Val MAE {:.5f}".format(test_mae * self.target_norm[1], val_mae * self.target_norm[1]))
        # load the best weights
        self.model.load_state_dict(torch.load(os.path.join(self.outpath, save_model)))
        return None

    def score_smiles(self, smi):
        # fixme I am not sure about the right way with transforms / prestransforms
        self.model.eval()
        try:
            graph = inputs._mol_to_graph(smi, float(0.0))
        except Exception as e:
            return None
        graph = self.transform(graph)
        dataLoader = DataLoader([graph], batch_size=1, shuffle=False)
        norm_score = self.model(list(dataLoader)[0].to(device)).detach().cpu().numpy()[0]
        return norm_score


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    with open(args.config, 'r') as f:
        config = json.loads(f)

    env = Environment(cfg.target_norm, cfg.test_prob, cfg.valid_prob, cfg.b_size, cfg.outpath, config, cfg.load_model)
    env.load_dataset(cfg.db_root, cfg.file_names)
    env.train_model(cfg.num_epochs, cfg.model_name)
