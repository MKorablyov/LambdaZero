import os

import numpy as np
import torch
from ray import tune
from torch_geometric.data import DataLoader

from LambdaZero.examples.dataset_splitting import RandomDatasetSplitter


class Env3dModelTrainer(tune.Trainable):
    def _setup(self, config: dict):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # split dataset via random split
        assert (
            config.get("train_ratio", 0.8) + config.get("valid_ratio", 0.1) <= 1.0
        ), "Train and validation data ratio should be less than 1."

        dataset_splitter = RandomDatasetSplitter(train_fraction=config.get("train_ratio", 0.8),
                                                 validation_fraction=config.get("valid_ratio", 0.1),
                                                 random_seed=config.get("seed_for_dataset_split", 0))

        dataset = self.config["dataset"](**self.config["dataset_config"])
        training_dataset, validation_dataset, test_dataset = dataset_splitter.get_split_datasets(dataset)

        batchsize = config.get("batchsize", 32)

        self.train_set = DataLoader(training_dataset, shuffle=True, batch_size=batchsize)
        self.val_set = DataLoader(validation_dataset, batch_size=batchsize)
        self.test_set = DataLoader(test_dataset, batch_size=batchsize)

        # make model
        self.model = config["model"](**config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](
            self.model.parameters(), **config["optimizer_config"]
        )
        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

        # for early stopping based on the validation loss
        self.best_valid_loss = np.inf
        self.patience = config["patience"]
        self.epoch_since_best_valid = 0

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
        
        # check if validation loss is getting worse
        validation_loss = scores["eval_loss"]
        
        # if not, then register this new better validation loss
        if validation_loss < self.best_valid_loss:
            self.best_valid_loss = validation_loss
            self.epoch_since_best_valid = 0
        # if worse, then look at patience to see if we stop the run
        else:
            self.epoch_since_best_valid += 1
        
        # trigger early stopping when patience runs out
        scores["early_stopping"] = self.epoch_since_best_valid >= self.patience

        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
