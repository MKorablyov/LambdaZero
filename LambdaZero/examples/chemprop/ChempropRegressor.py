from ray.tune import tune
import os.path
import torch
from torch import tensor
import numpy as np
from torch.optim import Adam
from torch_geometric.data import DataLoader

import LambdaZero.inputs
from LambdaZero.representation_learning.models.chemprop_model import ChempropNet


class ChempropRegressor(tune.Trainable):
    def _setup(self, config):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load dataset
        dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                               props=config["molprops"],
                                               transform=config["transform"],
                                               file_names=config["file_names"])

        # split dataset
        split_path = os.path.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
        train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)
        self.train_set = DataLoader(dataset[tensor(train_idxs)], shuffle=True, batch_size=config["b_size"])
        self.val_set = DataLoader(dataset[tensor(val_idxs)], batch_size=config["b_size"])
        self.test_set = DataLoader(dataset[tensor(test_idxs)], batch_size=config["b_size"])

        # make model
        self.model = ChempropNet(**config["model_parameters"])

        self.model.to(self.device)
        self.optim = Adam(self.model.parameters(), lr=config["lr"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.train_set, self.model,  self.device, self.config)
        # rename to make scope
        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
