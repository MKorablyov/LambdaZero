import sys
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Callable
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from LambdaZero.datasets.brutal_dock.mlflow_logger import MLFlowLogger


class AbstractModelTrainer(ABC):
    """
    This abstract base class must be derived to implement a specific
    model stepper. It then trains a pytorch model based on specified parameters.
    """
    optimizer_class = Adam

    train_loss_key = 'train_loss'
    validation_loss_key = 'val_loss'
    epoch_key = 'epoch'

    def __init__(self, loss_function: Callable[[torch.tensor, torch.tensor], torch.tensor],
                 device:  torch.device,
                 mlflow_logger: MLFlowLogger, score_mean: float = 0, score_std: float = 1.):

        self.loss_function = loss_function
        self.device = device
        self.mlflow_logger = mlflow_logger

        self.score_mean = torch.tensor(score_mean, device=device, requires_grad=False)
        self.score_std = torch.tensor(score_std, device=device, requires_grad=False)

    def _normalize_target(self, y):
        return (y - self.score_mean)/self.score_std

    def _denormalize_target(self, normalized_y):
        return self.score_std*normalized_y + self.score_mean

    @abstractmethod
    def _get_target_from_batch(self, batch):
        pass

    @abstractmethod
    def _apply_model_to_batch(self, batch, model):
        pass

    @abstractmethod
    def _get_size_of_batch(self, batch):
        pass

    def _get_batch_loss(self, batch, model):
        y_actual = self._get_target_from_batch(batch)
        normalized_y_actual = self._normalize_target(y_actual)

        normalized_y_predicted = self._apply_model_to_batch(batch, model)
        batch_loss = self.loss_function(normalized_y_actual, normalized_y_predicted)
        return batch_loss

    def _train_epoch(self, dataloader: DataLoader, model: nn.Module, optimizer, epoch: int):
        model.train()
        total_epoch_loss = 0.0

        number_of_batches = len(dataloader)

        for batch in tqdm(dataloader, desc="TRAIN", file=sys.stdout):
            optimizer.zero_grad()
            batch_loss = self._get_batch_loss(batch, model)
            batch_loss.backward()
            optimizer.step()

            batch_loss_value = batch_loss.item()

            self.mlflow_logger.increment_step_and_log_metrics(self.train_loss_key, batch_loss_value)
            self.mlflow_logger.log_metrics_at_current_step(self.epoch_key, epoch)

            total_epoch_loss += batch_loss_value*self._get_size_of_batch(batch)

        average_epoch_loss = total_epoch_loss/len(dataloader.dataset)
        return average_epoch_loss

    def _validation_epoch(self, dataloader: DataLoader, model: nn.Module):
        model.eval()
        total_epoch_loss = 0.0

        for batch in tqdm(dataloader, desc="VALID", file=sys.stdout):
            batch_loss = self._get_batch_loss(batch, model)
            batch_loss_value = batch_loss.item()
            total_epoch_loss += batch_loss_value*self._get_size_of_batch(batch)

        average_epoch_loss = total_epoch_loss/len(dataloader.dataset)

        self.mlflow_logger.log_metrics_at_current_step(self.validation_loss_key, average_epoch_loss)

        return average_epoch_loss

    def train_model(self, model: nn.Module, training_dataloader: DataLoader, validation_dataloader: DataLoader,
                    best_model_output_path: Path, num_epochs: int, patience: int,
                    lr=0.001, sched_factor=0.7, sched_patience=5, min_lr=0.00001):
        model.to(self.device)
        optimizer = self.optimizer_class(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=sched_factor,
                                                               patience=sched_patience,
                                                               min_lr=min_lr)

        best_validation_loss = float("inf")
        step_worse = 0
        for epoch in range(1, num_epochs + 1):

            lr = scheduler.optimizer.param_groups[0]['lr']
            average_training_loss = self._train_epoch(training_dataloader, model, optimizer, epoch)

            average_validation_loss = self._validation_epoch(validation_dataloader, model)

            scheduler.step(average_validation_loss)

            if average_validation_loss <= best_validation_loss:
                step_worse = 0
                best_validation_loss = average_validation_loss
                torch.save(model.state_dict(), best_model_output_path)
            else:
                step_worse += 1

            # early stopping
            if step_worse == patience:
                info = f"Early stopping point reached at epoch {epoch - patience:03d}"
                logging.info(info)
                model.load_state_dict(torch.load(best_model_output_path))
                break

            info = f"Epoch: {epoch:03d}, LR: {lr:5f}, Train Loss: {average_training_loss:.5f}, Val Loss: {average_validation_loss:.5f}"
            logging.info(info)

        return best_validation_loss

    def apply_model(self, model: nn.Module, dataloader: DataLoader):
        model.eval()

        list_actuals = []
        list_predicted = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="EVAL", file=sys.stdout):

                y_actual = self._get_target_from_batch(batch)
                list_actuals.extend(list(y_actual.cpu().numpy()))

                normalized_y_predicted = self._apply_model_to_batch(batch, model)
                y_predicted = self._denormalize_target(normalized_y_predicted)
                list_predicted.extend(list(y_predicted.cpu().numpy()))

        return np.array(list_actuals), np.array(list_predicted)


class XYModelTrainer(AbstractModelTrainer):

    def _get_xy_from_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def _get_target_from_batch(self, batch):
        _, y = self._get_xy_from_batch(batch)
        return y

    def _apply_model_to_batch(self, batch, model):
        x, _ = self._get_xy_from_batch(batch)
        y_hat = model.forward(x)
        return y_hat

    def _get_size_of_batch(self, batch):
        x, y = self._get_xy_from_batch(batch)
        return len(x)


class MoleculeModelTrainer(AbstractModelTrainer):

    def _get_target_from_batch(self, batch):
        batch = batch.to(self.device)
        return batch.gridscore

    def _apply_model_to_batch(self, batch, model):
        model.to(self.device)
        batch = batch.to(self.device)
        y_hat = model.forward(batch)
        return y_hat

    def _get_size_of_batch(self, batch):
        return batch.num_graphs
