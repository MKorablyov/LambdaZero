from typing import Dict

from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from LambdaZero.chemprop_adaptors.dataloader_utils import chemprop_collate_fn
from LambdaZero.datasets.dataset_splitting import RandomDatasetSplitter
from LambdaZero.examples.chemprop.datasets import D4ChempropMoleculesDataset
from LambdaZero.loggers.wandb_logger import RayTuneWandbLogger
from LambdaZero.oracle_models.chemprop_model import MolGraphChempropNet
from LambdaZero.trainable.base_pytorch_regressor import BasePytorchRegressor
import torch
import numpy as np


class ChempropRegressor(BasePytorchRegressor):

    loss = torch.nn.MSELoss()

    def get_model(self, config):
        return MolGraphChempropNet(**config["model_parameters"])

    def setup_logger(self, config):
        self.logger = RayTuneWandbLogger(
            config=config, log_dir=self.logdir, trial_id=self.trial_id
        )

    def _get_mean_and_std(self, config):
        mean = torch.tensor(
            config["target_norm"][0], device=self.device, requires_grad=False
        )
        std = torch.tensor(
            config["target_norm"][1], device=self.device, requires_grad=False
        )
        return mean, std

    @staticmethod
    def _get_logging_metrics(average_epoch_loss: float, config: Dict):
        std = config["target_norm"][1]
        rmse = std * np.sqrt(average_epoch_loss)

        return {"mse_normalized_units": average_epoch_loss, "rmse_original_units": rmse}

    @staticmethod
    def _normalize_target(y, mean, std):
        return (y - mean) / std

    @staticmethod
    def _denormalize_target(normalized_y, mean, std):
        return std * normalized_y + mean

    @staticmethod
    def _get_size_of_batch(batch: Dict, config):
        return len(batch[config["target"]])

    def _get_batch_loss(self, batch: Dict, model, config):
        mean, std = self._get_mean_and_std(config)

        batch_mol_graph = batch["mol_graph"]
        y_actual = batch[config["target"]].to(self.device)

        normalized_y_actual = self._normalize_target(y_actual, mean, std)

        model.to(self.device)
        normalized_y_predicted = model.forward(batch_mol_graph)

        batch_loss = self.loss(normalized_y_actual, normalized_y_predicted)
        return batch_loss

    def train_epoch(self, training_dataloader, model, optim, device, config):
        model.train()
        total_epoch_loss = 0.0

        for batch in tqdm(training_dataloader, desc="TRAIN", file=sys.stdout):
            self.optim.zero_grad()
            batch_loss = self._get_batch_loss(batch, model, config)
            batch_loss.backward()
            self.optim.step()

            batch_loss_value = batch_loss.item()

            total_epoch_loss += batch_loss_value * self._get_size_of_batch(
                batch, config
            )

        average_epoch_loss = total_epoch_loss / len(training_dataloader.dataset)
        return self._get_logging_metrics(average_epoch_loss, config)

    def eval_epoch(self, validation_dataloader, model, device, config):
        model.eval()
        total_epoch_loss = 0.0

        for batch in tqdm(validation_dataloader, desc="VALID", file=sys.stdout):
            batch_loss = self._get_batch_loss(batch, model, config)
            batch_loss_value = batch_loss.item()
            total_epoch_loss += batch_loss_value * self._get_size_of_batch(
                batch, config
            )

        average_epoch_loss = total_epoch_loss / len(validation_dataloader.dataset)
        return self._get_logging_metrics(average_epoch_loss, config)

    def get_dataset(self, config):
        dataset = D4ChempropMoleculesDataset(config["dataset_root"])
        return dataset

    def get_dataloaders(self, config):
        dataset = self.get_dataset(config)

        splitter = RandomDatasetSplitter(
            train_fraction=0.8,
            validation_fraction=0.1,
            random_seed=config["random_seed"],
        )

        training_dataset, validation_dataset, test_dataset = splitter.get_split_datasets(
            dataset
        )

        batch_size = config["batch_size"]

        training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            collate_fn=chemprop_collate_fn,
            shuffle=True,
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            collate_fn=chemprop_collate_fn,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=chemprop_collate_fn,
            shuffle=False,
        )

        return training_dataloader, validation_dataloader, test_dataloader
