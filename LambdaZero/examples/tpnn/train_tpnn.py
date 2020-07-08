import os
import numpy as np
import ray
from ray import tune
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F

import LambdaZero.inputs
import LambdaZero.models
import LambdaZero.utils
from LambdaZero.utils import get_external_dirs

from torch_scatter import scatter
from e3nn.networks import GatedConvNetwork
from e3nn.point.message_passing import Convolution

class MeanNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = GatedConvNetwork(*args, **kwargs)

    def forward(self, data):
        output = self.network(data.x, data.edge_index, data.edge_attr, size=data.x.shape[0])
        return scatter(output, data.batch, dim=0, reduce="mean")


def train_epoch(loader, model, optimizer, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.train()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for bidx, data in enumerate(loader):
        # compute y_hat and y
        data = data.to(device)

        optimizer.zero_grad()
        logit = model(data)
        pred = (logit * target_norm[1]) + target_norm[0]
        y = data.y

        loss = F.l1_loss(logit, ((y - target_norm[0]) / target_norm[1]).view(-1, 1))
        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item() * data.num_graphs
        metrics["mse"] += ((y - pred) ** 2).sum().item()
        metrics["mae"] += ((y - pred).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    return metrics


def eval_epoch(loader, model, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    with torch.no_grad():
        for bidx, data in enumerate(loader):
            # compute y_hat and y
            data = data.to(device)
            logit = model(data)
            pred = (logit * target_norm[1]) + target_norm[0]
            y = data.y

            loss = F.l1_loss(logit, ((y - target_norm[0]) / target_norm[1]).view(-1, 1))
            metrics["loss"] += loss.item() * data.num_graphs
            metrics["mse"] += ((y - pred) ** 2).sum().item()
            metrics["mae"] += ((y - pred).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(loader.dataset)
    metrics["mse"] = metrics["mse"] / len(loader.dataset)
    metrics["mae"] = metrics["mae"] / len(loader.dataset)
    return metrics


class TPNNregressor(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(torch.float64)

        # load dataset
        dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                               props=config["molprops"],
                                               transform=config["transform"],
                                               file_names=config["file_names"],
                                               proc_func=LambdaZero.inputs.tpnn_proc)

        # split dataset
        split_path = os.path.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
        train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)

        self.train_set = DataLoader(dataset[torch.tensor(train_idxs)], shuffle=True, batch_size=config["b_size"])
        self.val_set = DataLoader(dataset[torch.tensor(val_idxs)], batch_size=config["b_size"])
        self.test_set = DataLoader(dataset[torch.tensor(test_idxs)], batch_size=config["b_size"])

        # make model
        self.model = MeanNetwork(config["model_params"]["Rs_in"], 
                                 config["model_params"]["Rs_hidden"], 
                                 config["model_params"]["Rs_out"], 
                                 config["model_params"]["lmax"], 
                                 config["model_params"]["layers"], 
                                 config["model_params"]["max_radius"], 
                                 convolution=Convolution, 
                                 min_radius=config["model_params"]["min_radius"])
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.val_set, self.model, self.device, self.config)
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


transform = LambdaZero.inputs.tpnn_transform
datasets_dir, programs_dir, summaries_dir = get_external_dirs()


TPNN_CONFIG = {
    "trainer": TPNNregressor,
    "trainer_config": {
        "dataset_root": os.path.join(datasets_dir, "brutal_dock", "mpro_6lze"),
        "targets": ["gridscore"],
        "target_norms": [[-49.3, 26.1]],  # median, q(0.99) - q(0.01) 
        "file_names": ['Zinc15_260k_0', 'Zinc15_260k_1', 'Zinc15_260k_2', 'Zinc15_260k_3'],
        "transform": transform,
        "split_name": "randsplit_zinc15_260k",
        "lr": 0.001,
        "b_size": 16,
        "dim": 64,
        "num_epochs": 10,
        "model_params": {
		"Rs_in": [(23, 0)],
		"Rs_hidden": [(16, 0), (16, 1)],
		"Rs_out": [(1, 0)],
		"lmax": 3,
		"layers": 3,
		"max_radius": 2.26,
		"min_radius": 1.08	
	},
        "molprops": ["gridscore", "coord"],
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        # todo: test epoch
    },
    "summaries_dir": summaries_dir,
    "memory": 6 * 10 ** 9,
    "checkpoint_freq": 250000000,
    "stop": {"training_iteration": 2},
}

config = TPNN_CONFIG


if __name__ == "__main__":
    ray.init()

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop={"training_iteration": config["trainer_config"]["num_epochs"]},
                        resources_per_trial={
                            "cpu": 11,
                            "gpu": 1},
                        num_samples=1,
                        checkpoint_at_end=False,
                        local_dir=summaries_dir,
                        checkpoint_freq=100000)
    ray.shutdown()