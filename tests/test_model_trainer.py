import logging
import tempfile
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from LambdaZero.datasets.temp_brunos_work.loggers.mlflow_logger import MLFlowLogger
from LambdaZero.datasets.temp_brunos_work.model_trainer import XYModelTrainer
from tests.fake_linear_dataset import FakeLinearDataset
from tests.linear_regression import LinearRegression


@pytest.fixture
def number_of_points():
    return 1000


@pytest.fixture
def number_of_inputs():
    return 4


@pytest.fixture
def number_of_outputs():
    return 1


@pytest.fixture
def batch_size():
    return 250


@pytest.fixture
def best_model_path():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str).joinpath("test_best_model_path")
    logging.info("deleting test folder")


@pytest.fixture
def linear_dataset(number_of_points, number_of_inputs, number_of_outputs, batch_size):
    seed = 123
    dataset = FakeLinearDataset(seed, number_of_points, number_of_inputs, number_of_outputs)
    return dataset


@pytest.fixture
def starting_state_dict(linear_dataset):

    torch.manual_seed(23423)

    noise_amplitude = 0.1
    w = linear_dataset.weights.transpose(1, 0)
    b = linear_dataset.bias

    w_noise = noise_amplitude*torch.rand(w.shape)
    b_noise = noise_amplitude*torch.rand(b.shape)

    state_dict = OrderedDict([('linear.weight', w+w_noise), ('linear.bias', b+b_noise)])

    return state_dict


@pytest.fixture
def model(starting_state_dict, number_of_inputs, number_of_outputs):
    linear_model = LinearRegression(number_of_inputs, number_of_outputs)
    linear_model.load_state_dict(starting_state_dict)

    return linear_model


@pytest.fixture
def dataloaders(linear_dataset, number_of_points, batch_size):
    training_dimension = int(0.8 * len(linear_dataset))
    validation_dimension = number_of_points-training_dimension

    train_data, val_data = torch.utils.data.random_split(linear_dataset, [training_dimension, validation_dimension])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
    valid_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=0)

    return train_dataloader, valid_dataloader


@pytest.fixture
def mlflow_logger(experiment_name, tracking_uri):
    tags = {'model_type': 'trivial_linear_regresssion'}
    run_parameters = {'experiment_name': experiment_name, 'run_name': 'TEST'}
    mlflow_logger = MLFlowLogger(run_parameters, tracking_uri, tags)
    return mlflow_logger


def test_xy_model_trainer(mlflow_logger, dataloaders, model, best_model_path):

    training_dataloader, validation_dataloader = dataloaders

    loss_function = F.mse_loss
    device = torch.device('cpu')

    model_trainer = XYModelTrainer(loss_function, device, mlflow_logger)

    _ = model_trainer.train_model(model,
                                  training_dataloader,
                                  validation_dataloader,
                                  best_model_path,
                                  num_epochs=100)

    train_loss = [x.value for x in mlflow_logger.mlflow_client.get_metric_history(mlflow_logger.run_id, 'train_loss')]
    val_loss = [x.value for x in mlflow_logger.mlflow_client.get_metric_history(mlflow_logger.run_id, 'val_loss')]

    train_ratio = train_loss[-1]/train_loss[0]
    val_ratio = val_loss[-1]/val_loss[0]

    assert train_ratio < 1e-2, "training error is still large"
    assert val_ratio < 1e-2, "validation error is still large"


def test_xy_model_trainer_get_size_of_batch(mlflow_logger, dataloaders, model, batch_size):

    training_dataloader, validation_dataloader = dataloaders

    loss_function = F.mse_loss
    device = torch.device('cpu')

    model_trainer = XYModelTrainer(loss_function, device, mlflow_logger)
    batch = iter(training_dataloader).next()

    computed_batch_size = model_trainer._get_size_of_batch(batch)

    assert computed_batch_size == batch_size

