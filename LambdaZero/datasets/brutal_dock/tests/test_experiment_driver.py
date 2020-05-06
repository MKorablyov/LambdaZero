import logging
import tempfile
from pathlib import Path

import pytest
from torch_geometric.data import InMemoryDataset

from LambdaZero.core.alpha_zero_policy import torch
from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.models import MessagePassingNet
from LambdaZero.datasets.brutal_dock.parameter_inputs import RUN_PARAMETERS_KEY, TRAINING_PARAMETERS_KEY, \
    MODEL_PARAMETERS_KEY


@pytest.fixture
def expected_raw_file():
    return 'dock_blocks105_walk40_clust.feather'


@pytest.fixture
def expected_processed_file():
    return 'd4_processed_data.pt'


@pytest.fixture
def data_dir(expected_raw_file):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        data_dir_path = Path(tmp_dir_str).joinpath('raw/')
        data_dir_path.mkdir(exist_ok=True)
        data_dir_path.joinpath(expected_raw_file).touch(mode=0o666, exist_ok=True)
        yield data_dir_path
    logging.info("deleting test folder")


@pytest.fixture
def work_dir(expected_raw_file, expected_processed_file, random_molecule_dataset):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        raw_path = Path(tmp_dir_str).joinpath('raw/')
        raw_path.mkdir(exist_ok=True)
        raw_path.joinpath(expected_raw_file).touch(mode=0o666, exist_ok=True)

        processed_path = Path(tmp_dir_str).joinpath('processed/')
        processed_path.mkdir(exist_ok=True)
        processed_file_path = processed_path.joinpath(expected_processed_file)

        # This is a bit dirty. The collate method should really be a class method;
        # looking at that code, there is no reference to "self" internally. Here
        # we just want to spoof what this method does so we can shove our dataset
        # in the right place.
        torch.save(InMemoryDataset.collate(InMemoryDataset, random_molecule_dataset), str(processed_file_path))

        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def config(data_dir, work_dir, output_dir, number_of_node_features):
    run_parameters = dict(data_directory=str(data_dir),
                          working_directory=str(work_dir),
                          output_directory=str(output_dir),
                          tracking_uri=str(output_dir.joinpath("mlruns")),
                          experiment_name="TEST",
                          run_name="exp-driver-smoke-test",
                          git_hash="SOMETESTHASH",
                          user="test-user",
                          execution_file_name="test_file_name.py"
                          )

    training_parameters = dict(num_epochs=10,
                               num_workers=0,
                               batch_size=2,
                               learning_rate=1e-3,
                               train_fraction=0.8,
                               validation_fraction=0.1)

    model_parameters = dict(name="MPNN",
                            node_feat=number_of_node_features,
                            edge_feat=4,
                            gcn_size=8,
                            edge_hidden=8,
                            gru_out=8,
                            gru_layers=1,
                            linear_hidden=8)

    config = {RUN_PARAMETERS_KEY: run_parameters,
              TRAINING_PARAMETERS_KEY: training_parameters,
              MODEL_PARAMETERS_KEY: model_parameters}

    return config


@pytest.mark.parametrize("number_of_molecules", [20])
def test_smoke_test_experiment_driver(config):
    dataset_class = D4MoleculesDataset
    model_class = MessagePassingNet
    _ = experiment_driver(config, dataset_class, model_class)