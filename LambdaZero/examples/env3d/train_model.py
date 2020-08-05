import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import ray
from ray import tune
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import LambdaZero
from LambdaZero.examples.env3d.dataset import ENV3D_DATA_PROPERTIES
from LambdaZero.examples.env3d.dataset.processing import env3d_proc, transform_concatenate_positions_to_node_features
from LambdaZero.inputs import BrutalDock
from LambdaZero.utils import get_external_dirs


class Env3dModelTrainer(tune.Trainable):
    def _setup(self, config: dict):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # split dataset via random split
        assert (
            config.get("train_ratio", 0.8) + config.get("valid_ratio", 0.1) <= 1.0
        ), "Train and validation data ratio should be less than 1."
        np.random.seed(config.get("seed_for_dataset_split", 0))
        ndata = len(config["dataset"])
        shuffle_idx = np.random.shuffle(np.arange(ndata))
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
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

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


datasets_dir, _, summaries_dir = get_external_dirs()
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")

data_filename_without_suffix = "env3d_dataset_5_parent_blocks"
data_filename = f"{data_filename_without_suffix}.feather"

source_path_to_dataset = results_dir.joinpath(data_filename)

# from train_mpnn
# to do: clean up + docstring
def train_epoch(loader, model, optimizer, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss": 0}
    epoch_targets = []
    epoch_preds = []

    for data in loader:
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        logits = model(data)
        loss = F.mse_loss(logits, normalizer.forward_transform(targets))
        loss.backward()
        optimizer.step()

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.backward_transform(logits).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets, 0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds) ** 2).mean()

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(
        ranked_targets[:15]
    )
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(
        ranked_targets[:50]
    )
    return metrics


def eval_epoch(loader, model, device, config):
    normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
    model.eval()
    metrics = {"loss": 0}
    epoch_targets = []
    epoch_preds = []

    for data in loader:
        data = data.to(device)
        targets = getattr(data, config["target"])

        logits = model(data)
        loss = F.mse_loss(logits, normalizer.forward_transform(targets))

        # log stuff
        metrics["loss"] += loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(normalizer.backward_transform(logits).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets, 0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds) ** 2).mean()

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(
        ranked_targets[:15]
    )
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(
        ranked_targets[:50]
    )
    return metrics


if __name__ == "__main__":

    ray.init(local_mode=True)

    with tempfile.TemporaryDirectory() as root_directory:
        raw_data_directory = Path(root_directory).joinpath("raw")
        raw_data_directory.mkdir()
        dest_path_to_dataset = raw_data_directory.joinpath(data_filename)
        shutil.copyfile(source_path_to_dataset, dest_path_to_dataset)

        dataset = BrutalDock(
            root_directory,
            props=ENV3D_DATA_PROPERTIES,
            file_names=[data_filename_without_suffix],
            proc_func=env3d_proc,
            transform=transform_concatenate_positions_to_node_features,
        )

        print(f"size of dataset: {len(dataset)}")

    # TO DO: read hyperparameters from a config file as ray tune variables
    env3d_config = {
        "trainer": Env3dModelTrainer,
        "trainer_config": {
            "dataset": dataset,
            "seed_for_dataset_split": 0,
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "batchsize": 32,
            "model": None,
            "model_config": {},
            "optimizer": torch.optim.Adam,
            "optimizer_config": {"lr": 1e-3},
            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
            "target": "gridscore",
            "target_norm": [-43.042, 7.057],
        },
        "summaries_dir": summaries_dir,
        "memory": 10 * 10 ** 9,
        "stop": {"training_iteration": 200},
        "resources_per_trial": {
            "cpu": 4,  # fixme - calling ray.remote would request resources outside of tune allocation
            "gpu": 1.0,
        },
        "keep_checkpoint_num": 2,
        "checkpoint_score_attr": "train_loss",
        "num_samples": 1,
        "checkpoint_at_end": False,
    }

    ray.init(memory=env3d_config["memory"])

    analysis = tune.run(
        env3d_config["trainer"],
        config=env3d_config["trainer_config"],
        stop=env3d_config["stop"],
        resources_per_trial=env3d_config["resources_per_trial"],
        num_samples=env3d_config["num_samples"],
        checkpoint_at_end=env3d_config["checkpoint_at_end"],
        local_dir=summaries_dir,
        checkpoint_freq=env3d_config["checkpoint_freq"],
    )
