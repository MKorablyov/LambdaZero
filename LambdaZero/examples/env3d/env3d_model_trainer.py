import os

import numpy as np
import torch
from ray import tune
from torch_geometric.data import DataLoader


class Env3dModelTrainer(tune.Trainable):
    def _setup(self, config: dict):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # split dataset via random split
        assert (
            config.get("train_ratio", 0.8) + config.get("valid_ratio", 0.1) <= 1.0
        ), "Train and validation data ratio should be less than 1."
        np.random.seed(config.get("seed_for_dataset_split", 0))

        dataset = self.config["dataset"](**self.config["dataset_config"])
        ndata = len(dataset)
        shuffle_idx = np.arange(ndata)
        np.random.shuffle(shuffle_idx)
        n_train = int(config.get("train_ratio", 0.8) * ndata)
        n_valid = int(config.get("valid_ratio", 0.1) * ndata)
        train_idxs = shuffle_idx[:n_train]
        val_idxs = shuffle_idx[n_train : n_train + n_valid]
        test_idxs = shuffle_idx[n_valid:]
        batchsize = config.get("batchsize", 32)

        self.train_set = DataLoader(
            dataset[torch.tensor(train_idxs)], shuffle=True, batch_size=batchsize
        )
        self.val_set = DataLoader(dataset[torch.tensor(val_idxs)], batch_size=batchsize)
        self.test_set = DataLoader(
            dataset[torch.tensor(test_idxs)], batch_size=batchsize
        )

        # make model
        self.model = config["model"](**config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](
            self.model.parameters(), **config["optimizer_config"]
        )

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(
            self.train_set, self.model, self.optim, self.device, self.config
        )
        eval_scores = self.eval_epoch(
            self.train_set, self.model, self.device, self.config
        )
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