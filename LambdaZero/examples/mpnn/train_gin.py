import os

import numpy as np
import ray
from ray import tune
import torch
from torch_geometric.data import DataLoader

from LambdaZero.examples.mpnn.train_mpnn import DEFAULT_CONFIG
import LambdaZero.inputs
import LambdaZero.models
import LambdaZero.utils
from LambdaZero.utils import get_external_dirs


class GINRegressor(tune.Trainable):
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
        self.train_set = DataLoader(dataset[torch.tensor(train_idxs)], shuffle=True, batch_size=config["b_size"])
        self.val_set = DataLoader(dataset[torch.tensor(val_idxs)], batch_size=config["b_size"])
        self.test_set = DataLoader(dataset[torch.tensor(test_idxs)], batch_size=config["b_size"])

        # make model
        self.model = LambdaZero.models.GraphIsomorphismNet()
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.train_set, self.model, self.device, self.config)
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


_, _, summaries_dir = get_external_dirs()

DEFAULT_CONFIG["trainer"] = GINRegressor

config = DEFAULT_CONFIG


if __name__ == "__main__":
    ray.init()

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop={"training_iteration": 100},  # EarlyStop(),
                        resources_per_trial={
                            "cpu": 4,  # fixme requesting all CPUs blocks additional call to ray from LambdaZero.input
                            "gpu": 1.0},
                        num_samples=1,
                        checkpoint_at_end=False,
                        local_dir=summaries_dir,
                        checkpoint_freq=100000)
