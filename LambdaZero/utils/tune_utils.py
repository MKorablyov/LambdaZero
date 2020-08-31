import os.path as osp
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch.utils.data import Subset

from ray import tune


class BasicRegressor(tune.Trainable):
    def _setup(self, config):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # make dataset
        dataset = self.config["dataset"](**self.config["dataset_config"])
        train_idxs, val_idxs, _ = np.load(self.config["dataset_split_path"], allow_pickle=True)
        # fixme!!!!!!!!!!! --- make mpnn index --- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        train_idxs = train_idxs[train_idxs < len(dataset)]
        val_idxs = val_idxs[val_idxs < len(dataset)]
        self.train_set = Subset(dataset, train_idxs.tolist())
        self.val_set = Subset(dataset, val_idxs.tolist())


        self.train_loader = DataLoader(self.train_set,shuffle=True, batch_size=self.config["b_size"])
        self.val_loader = DataLoader(self.val_set, batch_size=self.config["b_size"])

        # make model
        self.model = config["model"](**config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](self.model.parameters(), **config["optimizer_config"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_loader, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.val_loader, self.model,  self.device, self.config)
        # rename to make scope
        train_scores = [("train_" + k, v) for k,v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = osp.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))