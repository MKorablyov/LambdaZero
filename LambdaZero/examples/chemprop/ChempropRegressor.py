from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from LambdaZero.chemprop_adaptors.dataloader_utils import chemprop_collate_fn
from LambdaZero.oracle_models.chemprop_model import MolGraphChempropNet
from LambdaZero.trainable.base_pytorch_regressor import BasePytorchRegressor
import torch


def normalize_target(y, mean, std):
    return (y - mean) / std


def denormalize_target(normalized_y, mean, std):
    return std * normalized_y + mean


class ChempropRegressor(BasePytorchRegressor):
    loss_function = torch.nn.functional.mse_loss

    def _get_batch_loss(self, batch, model, config):
        y_actual = self._get_target_from_batch(batch)
        normalized_y_actual = self._normalize_target(y_actual)

        normalized_y_predicted = self._apply_model_to_batch(batch, model)
        batch_loss = self.loss_function(normalized_y_actual, normalized_y_predicted)
        return batch_loss

    def train_epoch(self, training_dataloader, model, optim, device, config):
        model.train()
        total_epoch_loss = 0.0

        number_of_batches = len(training_dataloader)

        for batch in tqdm(training_dataloader, desc="TRAIN", file=sys.stdout):
            self.optim.zero_grad()
            batch_loss = self._get_batch_loss(batch, model)
            batch_loss.backward()
            self.optim.step()

            batch_loss_value = batch_loss.item()

            total_epoch_loss += batch_loss_value*self._get_size_of_batch(batch)

        average_epoch_loss = total_epoch_loss/len(dataloader.dataset)
        return average_epoch_loss

    def eval_epoch(self, validation_set, model, device, config):
        pass


    def get_dataloaders(self, config):

        batch_size = config["b_size"]

        training_dataloader = DataLoader(training_dataset,
                                         batch_size=batch_size,
                                         collate_fn=chemprop_collate_fn,
                                         shuffle=True)

        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=batch_size,
                                           collate_fn=chemprop_collate_fn,
                                           shuffle=True)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     collate_fn=chemprop_collate_fn,
                                     shuffle=False)

        return training_dataloader, validation_dataloader, test_dataloader


    def get_model(self, config):
        return MolGraphChempropNet(**config["model_parameters"])
