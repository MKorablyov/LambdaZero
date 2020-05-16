import logging
import tempfile
from pathlib import Path

import pytest
from torch_geometric.data import InMemoryDataset

from LambdaZero.core.alpha_zero_policy import torch
from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.loggers.mlflow_logger import MLFlowLogger
from LambdaZero.datasets.brutal_dock.models.chemprop_model import ChempropNet
from LambdaZero.datasets.brutal_dock.models.message_passing_model import MessagePassingNet
from LambdaZero.datasets.brutal_dock.parameter_inputs import RUN_PARAMETERS_KEY, TRAINING_PARAMETERS_KEY, \
    MODEL_PARAMETERS_KEY, TAGS_KEY, PATHS_KEY, CONFIG_KEY, NON_CONFIG_KEY


@pytest.fixture
def number_of_node_features(real_molecule_dataset):
    molecule_graph = real_molecule_dataset[0]
    node_features = molecule_graph.x
    number_features = node_features.shape[1]
    return number_features


@pytest.fixture
def expected_raw_files():
    return ['dock_blocks105_walk40_clust.feather', 'dock_blocks105_walk40_2_clust.feather']


@pytest.fixture
def expected_processed_file():
    return 'd4_processed_data.pt'


@pytest.fixture
def data_dir(expected_raw_files):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        data_dir_path = Path(tmp_dir_str).joinpath('raw/')
        data_dir_path.mkdir(exist_ok=True)
        for expected_raw_file in expected_raw_files:
            data_dir_path.joinpath(expected_raw_file).touch(mode=0o666, exist_ok=True)
        yield data_dir_path
    logging.info("deleting test folder")


@pytest.fixture
def work_dir(expected_raw_files, expected_processed_file, real_molecule_dataset):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        raw_path = Path(tmp_dir_str).joinpath('raw/')
        raw_path.mkdir(exist_ok=True)
        for expected_raw_file in expected_raw_files:
            raw_path.joinpath(expected_raw_file).touch(mode=0o666, exist_ok=True)

        processed_path = Path(tmp_dir_str).joinpath('processed/')
        processed_path.mkdir(exist_ok=True)
        processed_file_path = processed_path.joinpath(expected_processed_file)

        # This is a bit dirty. The collate method should really be a class method;
        # looking at that code, there is no reference to "self" internally. Here
        # we just want to spoof what this method does so we can shove our dataset
        # in the right place.
        torch.save(InMemoryDataset.collate(InMemoryDataset, real_molecule_dataset), str(processed_file_path))

        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def paths(data_dir, work_dir, output_dir):
    paths = dict(data_directory=str(data_dir),
                 working_directory=str(work_dir),
                 output_directory=str(output_dir),
                 tracking_uri=str(output_dir.joinpath("mlruns")))
    return paths


@pytest.fixture
def model_class(model_name):
    if model_name == "MPNN":
        return MessagePassingNet
    elif model_name == "chemprop":
        return ChempropNet


@pytest.fixture
def model_parameters(model_name, number_of_node_features):
    parameters = None
    if model_name == "MPNN":
        parameters = dict(name="MPNN",
                          node_feat=number_of_node_features,
                          edge_feat=4,
                          gcn_size=8,
                          edge_hidden=8,
                          gru_layers=1,
                          linear_hidden=8)

    elif model_name == "chemprop":
        parameters = dict(name="chemprop",
                          depth=2,
                          ffn_num_layers=2,
                          ffn_hidden_size=8
                          )
    return parameters


@pytest.fixture
def input_and_run_config(paths, model_parameters):

    tags = dict(git_hash="SOMETESTHASH",
                user="test-user",
                execution_file_name="test_file_name.py",
                run_name="exp-driver-smoke-test")

    run_parameters = dict(experiment_name="TEST",
                          run_name="exp-driver-smoke-test")

    training_parameters = dict(num_epochs=10,
                               num_workers=0,
                               batch_size=2,
                               learning_rate=1e-3,
                               train_fraction=0.8,
                               validation_fraction=0.1,
                               patience=1)

    config = {RUN_PARAMETERS_KEY: run_parameters,
              TRAINING_PARAMETERS_KEY: training_parameters,
              MODEL_PARAMETERS_KEY: model_parameters}

    non_config = {PATHS_KEY: paths,
                  TAGS_KEY: tags}

    config_and_augmented = {CONFIG_KEY: config,
                            NON_CONFIG_KEY: non_config}

    return config_and_augmented


@pytest.mark.parametrize("model_name", ["chemprop", "MPNN"])
def test_smoke_test_experiment_driver(model_name, model_class, input_and_run_config):
    dataset_class = D4MoleculesDataset
    logger_class = MLFlowLogger

    with pytest.warns(None) as record:
        _ = experiment_driver(input_and_run_config, dataset_class, model_class, logger_class)

    for warning in record.list:
        assert warning.category != UserWarning, "A user warning was raised"

