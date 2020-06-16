import sys
from pathlib import Path

import ray
from torch.nn.functional import mse_loss

from torch import nn
from ray import tune
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from LambdaZero import ROOT_DIR
from LambdaZero.representation_learning.dataloader_utils import get_geometric_dataloaders
from LambdaZero.representation_learning.dataset_utils import get_scores_statistics
from LambdaZero.representation_learning.datasets import D4GeometricMoleculesDataset
from LambdaZero.oracle_models.message_passing_model import MessagePassingNet
from LambdaZero.representation_learning.parameter_inputs import CONFIG_KEY, NON_CONFIG_KEY, PATHS_KEY, \
    TRAINING_PARAMETERS_KEY, MODEL_PARAMETERS_KEY, read_configuration_file, EXECUTION_FILENAME_KEY
from LambdaZero.utils import get_external_dirs


class ModelTrainer(tune.Trainable):
    optimizer_class = Adam

    @staticmethod
    def _get_model_and_parameters_from_configuration(config):
        model_class = config["model_class"]
        input_and_run_config = config["input_and_run_config"]
        model_parameters = input_and_run_config[CONFIG_KEY][MODEL_PARAMETERS_KEY]
        return model_class.create_model_for_training(model_parameters), model_parameters

    @staticmethod
    def _get_dataloaders_from_configuration(config):

        random_seed = config["random_seed"]

        dataset_class = config["dataset_class"]
        input_and_run_config = config["input_and_run_config"]

        paths_dict = input_and_run_config[NON_CONFIG_KEY][PATHS_KEY]
        training_parameters = input_and_run_config[CONFIG_KEY][TRAINING_PARAMETERS_KEY]

        get_dataloaders = config["get_dataloaders"]
        data_dir = Path(paths_dict["data_directory"])
        work_dir = Path(paths_dict["working_directory"])

        dataset = dataset_class.create_dataset(root_dir=work_dir,
                                               original_raw_data_dir=data_dir)

        training_dataloader, validation_dataloader, test_dataloader = \
            get_dataloaders(dataset, training_parameters, random_seed)

        return training_dataloader, validation_dataloader, test_dataloader

    @staticmethod
    def _get_optimiser(optimizer_class, config, model):

        input_and_run_config = config["input_and_run_config"]
        training_parameters = input_and_run_config[CONFIG_KEY][TRAINING_PARAMETERS_KEY]
        lr = training_parameters["learning_rate"]
        optimizer = optimizer_class(model.parameters(), lr=lr)
        return optimizer

    def _setup(self, config):
        self.loss_function = mse_loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.model_parameters = self._get_model_and_parameters_from_configuration(config)
        self.optimizer = self._get_optimiser(self.optimizer_class, config, self.model)

        self.training_dataloader, self.validation_dataloader, self.test_dataloader = \
            self._get_dataloaders_from_configuration(config)

        mean, std = get_scores_statistics(self.training_dataloader)

        self.score_mean = torch.tensor(mean, device=self.device, requires_grad=False)
        self.score_std = torch.tensor(std, device=self.device, requires_grad=False)

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

    def _normalize_target(self, y):
        return (y - self.score_mean)/self.score_std

    def _denormalize_target(self, normalized_y):
        return self.score_std*normalized_y + self.score_mean

    def _get_batch_loss(self, batch, model):
        y_actual = self._get_target_from_batch(batch)
        normalized_y_actual = self._normalize_target(y_actual)

        normalized_y_predicted = self._apply_model_to_batch(batch, model)
        batch_loss = self.loss_function(normalized_y_actual, normalized_y_predicted)
        return batch_loss

    def _train_epoch(self, dataloader: DataLoader, model: nn.Module):
        model.train()
        total_epoch_loss = 0.0

        for batch in tqdm(dataloader, desc="TRAIN", file=sys.stdout):
            self.optimizer.zero_grad()
            batch_loss = self._get_batch_loss(batch, model)
            batch_loss.backward()
            self.optimizer.step()

            batch_loss_value = batch_loss.item()

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

        return average_epoch_loss

    def _train(self):
        train_loss = self._train_epoch(self.training_dataloader, self.model)
        eval_loss = self._validation_epoch(self.validation_dataloader, self.model)
        scores = dict(train_loss=train_loss, eval_loss=eval_loss)
        return scores

    def _save(self, tmp_checkpoint_dir):
        torch.save(self.model.state_dict(), Path(tmp_checkpoint_dir).joinpath('model.pth'))

    def _restore(self, checkpoint):
        return model_class.load_model_from_file(checkpoint, self.model_parameters)


model_class = MessagePassingNet
dataset_class = D4GeometricMoleculesDataset

path_of_this_file = Path(__file__).resolve()

json_file_path = ROOT_DIR.joinpath("LambdaZero/examples/representation_learning_experiments/configurations/example_mpnn_input.json")

if __name__ == '__main__':

    datasets_dir, programs_dir, summaries_dir = get_external_dirs()

    config_parameters = read_configuration_file(json_file_path)

    debug_path = Path(summaries_dir).joinpath("debug")

    paths = {"tracking_uri": str(debug_path.joinpath("tracking_uri")),
             "working_directory": str(debug_path.joinpath("working")),
             "output_directory": str(debug_path.joinpath("output")),
             "data_directory": str(Path(datasets_dir).joinpath("brutal_dock/d4/raw/"))
             }

    non_config_parameters = {PATHS_KEY: paths,
                             EXECUTION_FILENAME_KEY: str(path_of_this_file.relative_to(ROOT_DIR))}

    input_and_run_config = {CONFIG_KEY: config_parameters,
                            NON_CONFIG_KEY: non_config_parameters}

    config = {"model_class": model_class,
              "dataset_class": dataset_class,
              "random_seed": 0,
              "get_dataloaders": get_geometric_dataloaders,
              "input_and_run_config": input_and_run_config}

    ray.init()
    analysis = tune.run(ModelTrainer,
                        config=config,
                        stop={"training_iteration": 1},
                        resources_per_trial={
                           "cpu": 4,
                           "gpu": 0
                        },
                        num_samples=1,
                        checkpoint_at_end=False,
                        local_dir=summaries_dir,
                        checkpoint_freq=100)
