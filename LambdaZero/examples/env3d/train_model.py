import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import ray
from ray import tune
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import LambdaZero
from LambdaZero.examples.env3d.dataset import ENV3D_DATA_PROPERTIES
from LambdaZero.examples.env3d.dataset.processing import env3d_proc, transform_concatenate_positions_to_node_features
from LambdaZero.inputs import BrutalDock
from LambdaZero.utils import get_external_dirs


NCLASS = 170


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


datasets_dir, _, summaries_dir = get_external_dirs()
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")

data_filename_without_suffix = "env3d_dataset_5_parent_blocks"
data_filename = f"{data_filename_without_suffix}.feather"

source_path_to_dataset = results_dir.joinpath(data_filename)

props = [
    "coord",
    "n_axis",
    "attachment_node_index",
    "attachment_angle",
    "attachment_block_index",
]


def class_and_angle_loss(block_predictions, block_targets, angle_predictions, angle_targets):
    """
    calculate losses for block predictions (class) and angle predictions

    Args:
        block_predictions (torch.Tensor): class predictions logits. size: (batchsize, number of classes)
        block_targets (torch.Tensor): class target as int. size (batchsize)
        angle_predictions (torch.Tensor): vector prediction for the angle. Will be converted to sin / cos
            size: (batchsize, 2)
        angle_targets (torch.Tensor): angle targets in radian between 0 and 2 pi. Value of -1 means an invalid angle.
            size: (batchsize)

    Returns:
        torch.Tensor: cross entropy for the class. size: (1,)
        torch.Tensor: mse for the sin and cos of the angle. size: (1,)
        torch.Tensor: mae for sin and cos of the angle. size (1,)
        int: number of valid angles
    """
    # prediction over classes in a straight-forward cross-entropy
    class_loss = F.cross_entropy(block_predictions, block_targets)

    # for the angle, first, we convert the outputs to sin / cos representation
    # sin = u / \sqrt{u² + v²}
    # cos = v / \sqrt{u² + v²}
    # get denominator
    norm = torch.norm(angle_predictions, dim=-1)
    # take max between norm and a small value to avoid division by zero
    norm = torch.max(norm, 1e-6 * torch.ones_like(norm))
    # norm is a (batchsize) tensor. convert to (batchsize, 2)
    norm = norm.unsqueeze(-1).repeat(1, 2)
    angle_predictions /= norm
    # angle_predictions[:, 0] is sin, [:, 1] is cos
    # now, convert the ground truth
    sin_target = torch.sin(angle_targets)
    cos_target = torch.cos(angle_targets)
    angle_target_sincos = torch.stack([sin_target, cos_target], dim=-1)

    # loss for the angle is the MSE
    # we want to calculate only when angle_predictions > -1
    angle_loss = F.mse_loss(angle_target_sincos, angle_predictions, reduction="none")
    angle_mae = F.l1_loss(angle_target_sincos, angle_predictions, reduction="none")

    # sum over last dimension, aka sin and cos
    angle_loss = torch.sum(angle_loss, dim=-1)
    angle_mae = torch.sum(angle_mae, dim=-1)

    # create a mask of 0 where angle_target is invalid (-1), and 1 elsewhere
    mask = torch.where(angle_targets > 0, torch.ones_like(angle_targets), torch.zeros_like(angle_targets))
    num_elem = torch.sum(mask)

    # calculate the mean over valid elements only
    angle_loss = torch.sum(angle_loss * mask) / max(num_elem, 1)
    angle_mae = torch.sum(angle_mae * mask) / max(num_elem, 1)

    return class_loss, angle_loss, angle_mae, num_elem


def train_epoch(loader, model, optimizer, device, config):
    """
    function training a model over an epoch. Metrics are computed as well

    Args:
        loader (torch_geometric.dataInMemoryDataset: training dataloader
        model (nn.Module): pytorch model
        optimizer (torch.optim): optimizer to train the model
        device (torch.device): device where the model / data will be processed
        config (dict): other parameters

    Returns:
        dict: metrics for the epoch. Includes:
            loss for block and angle predictions combined
            block_loss for block predictions only (cross-entropy)
            angle_loss for loss predictions only (MSE on sin and cos)
            accuracy for block predictions
            RMSE for sin/cos of angle
            MAE for sin/cos of angle
    """
    model.train()

    metrics = {}

    for data in loader:
        data = data.to(device)

        class_target = data.attachment_block_class
        angle_target = data.attachment_angle

        optimizer.zero_grad()
        # model outputs 2 tensors:
        # 1) size (batch, nclass) for block prediction
        # 2) size (batch, 2) for angle prediction. u and v should be combined to get sin / cos of angle
        class_predictions, angle_predictions = model(data)

        # prediction over classes in a straight-forward cross-entropy
        class_loss, angle_loss, angle_mae, n_angle = \
            class_and_angle_loss(class_predictions, class_target, angle_predictions, angle_target)

        loss = class_loss + config.get("angle_loss_weight", 1) * angle_loss

        loss.backward()
        optimizer.step()

        # log loss information
        # global cumulative losses on this epoch
        metrics["loss"] = metrics.get("loss", 0) + loss.item() * data.num_graphs
        metrics["block_loss"] = metrics.get("block_loss", 0) + class_loss.item() * data.num_graphs
        metrics["angle_loss"] = metrics.get("angle_loss", 0) + angle_loss.item() * data.num_graphs

        # let's log some more informative metrics as well
        # for prediction, accuracy is good to know
        _, predicted = torch.max(class_predictions, dim=1)
        correct = (predicted == class_target).sum().item()
        # consider the accuracy as a rolling average. We have to adjust the previously calculated accuracy by the number of elements in it
        metrics["block_accuracy"] = (metrics.get("block_accuracy", 0) * metrics.get("n_blocks", 0) + correct) / \
            max(1, metrics.get("n_blocks", 0) + data.num_graphs)
        metrics["n_blocks"] = metrics.get("n_blocks", 0) + data.num_graphs

        # for the angle, we want the RMSE and the MAE, both as rolling averages as well. Beware the mask for invalid angles here
        metrics["angle_rmse"] = (metrics.get("angle_rmse", 0) * metrics.get("n_angles", 0) + angle_loss.item() * n_angle) / \
            max(1, metrics.get("n_angles", 0) + n_angle)
        # we have to do the same for the MAE
        metrics["angle_mae"] = (metrics.get("angle_mae", 0) * metrics.get("n_angles", 0) + angle_mae.item()) / \
            max(1, metrics.get("n_angles", 0) + n_angle)

    # RMSE and MAE are possibly on gpu. Move to cpu and convert to numpy values (non-tensor)
    # also take the squareroot of the MSE to get the actual RMSE
    if device == torch.device("cpu"):
        metrics["angle_rmse"] = np.sqrt(metrics["angle_rmse"].numpy())
        metrics["angle_mae"] = metrics["angle_mae"].numpy()
    else:
        metrics["angle_rmse"] = np.sqrt(metrics["angle_rmse"].cpu().numpy())
        metrics["angle_mae"] = metrics["angle_mae"].cpu().numpy()

    return metrics


def eval_epoch(loader, model, device, config):
    """
    evaluate a model and return metrics

    Args:
        loader (torch_geometric.dataInMemoryDataset: training dataloader
        model (nn.Module): pytorch model
        device (torch.device): device where the model / data will be processed
        config (dict): other parameters

    Returns:
        dict: metrics for the epoch. Includes:
            loss for block and angle predictions combined
            block_loss for block predictions only (cross-entropy)
            angle_loss for loss predictions only (MSE on sin and cos)
            accuracy for block predictions
            RMSE for sin/cos of angle
            MAE for sin/cos of angle`
    """
    model.eval()
    metrics = {}

    for data in loader:
        data = data.to(device)

        class_target = data.attachment_block_index
        angle_target = data.attachment_angle

        # model outputs 2 tensors:
        # 1) size (batch, nclass) for block prediction
        # 2) size (batch, 2) for angle prediction. u and v should be combined to get sin / cos of angle
        class_predictions, angle_predictions = model(data)

        # prediction over classes in a straight-forward cross-entropy
        class_loss, angle_loss, angle_mae, n_angle = class_and_angle_loss(class_predictions, class_target, angle_predictions, angle_target)

        loss = class_loss + config.get("angle_loss_weight", 1) * angle_loss

        # log loss information
        # global cumulative losses on this epoch
        metrics["loss"] = metrics.get("loss", 0) + loss.item() * data.num_graphs
        metrics["block_loss"] = metrics.get("block_loss", 0) + class_loss.item() * data.num_graphs
        metrics["angle_loss"] = metrics.get("angle_loss", 0) + angle_loss.item() * data.num_graphs

        # let's log some more informative metrics as well
        # for prediction, accuracy is good to know
        _, predicted = torch.max(class_predictions, dim=1)
        correct = (predicted == class_target).sum().item()
        # consider the accuracy as a rolling average. We have to adjust the previously calculated accuracy by the number of elements in it
        metrics["block_accuracy"] = (metrics.get("block_accuracy", 0) * metrics.get("n_blocks", 0) + correct) / \
            max(1, metrics.get("n_blocks", 0) + data.num_graphs)
        metrics["n_blocks"] = metrics.get("n_blocks", 0) + data.num_graphs

        # for the angle, we want the RMSE and the MAE, both as rolling averages as well. Beware the mask for invalid angles here
        metrics["angle_rmse"] = (metrics.get("angle_rmse", 0) * metrics.get("n_angles", 0) + angle_loss.item() * n_angle) / \
            max(1, metrics.get("n_angles", 0) + n_angle)
        # we have to do the same for the MAE
        metrics["angle_mae"] = (metrics.get("angle_mae", 0) * metrics.get("n_angles", 0) + angle_mae.item()) / \
            max(1, metrics.get("n_angles", 0) + n_angle)

    # RMSE and MAE are possibly on gpu. Move to cpu and convert to numpy values (non-tensor)
    # also take the squareroot of the MSE to get the actual RMSE
    if device == torch.device("cpu"):
        metrics["angle_rmse"] = np.sqrt(metrics["angle_rmse"].numpy())
        metrics["angle_mae"] = metrics["angle_mae"].numpy()
    else:
        metrics["angle_rmse"] = np.sqrt(metrics["angle_rmse"].cpu().numpy())
        metrics["angle_mae"] = metrics["angle_mae"].cpu().numpy()

    return metrics


class DebugModel(nn.Module):
    """
    a model for dev & debug. Delete when real models are available.
    """
    def __init__(self):
        super(DebugModel, self).__init__()
        self.lin0 = nn.Linear(3, NCLASS)

    def forward(self, data):
        batchsize = data.num_graphs
        return self.lin0(torch.zeros([batchsize, 3]).to(torch.device('cuda'))), torch.ones([batchsize, 2]).to(torch.device('cuda'))


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
            "batchsize": 5,
            "model": DebugModel,  # to do: insert a real model here
            "model_config": {},
            "optimizer": torch.optim.Adam,
            "optimizer_config": {"lr": 1e-3},
            "angle_loss_weight": 1,
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

    # ray.init(memory=env3d_config["memory"])

    analysis = tune.run(
        env3d_config["trainer"],
        config=env3d_config["trainer_config"],
        stop=env3d_config["stop"],
        resources_per_trial=env3d_config["resources_per_trial"],
        num_samples=env3d_config["num_samples"],
        checkpoint_at_end=env3d_config["checkpoint_at_end"],
        local_dir=summaries_dir,
        checkpoint_freq=env3d_config.get("checkpoint_freq", 1),
    )
