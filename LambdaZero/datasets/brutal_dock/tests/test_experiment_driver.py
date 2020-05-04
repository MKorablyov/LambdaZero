import logging
import tempfile
from pathlib import Path

import pytest

from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.parameter_inputs import RUN_PARAMETERS_KEY, TRAINING_PARAMETERS_KEY, \
    MODEL_PARAMETERS_KEY


@pytest.fixture
def data_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def work_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def config(data_dir, work_dir, output_dir):
    run_parameters = dict(data_directory=str(data_dir),
                          working_directory=str(work_dir),
                          output_directory=str(output_dir),
                          num_workers=0,
                          tracking_uri=str(output_dir.joinpath("mlrun")),
                          experiment_name="TEST",
                          run_name="exp-driver-smoke-test",
                          git_hash="SOMETESTHASH",
                          user="test-user",
                          execution_file_name="test_file_name.py"
                          )

    training_parameters = dict(num_epochs=10,
                               batch_size=2,
                               learning_rate=1e-3,
                               train_fraction=0.8,
                               validation_fraction=0.1)

    model_parameters = dict(name="MPNN",
                            gcn_size=128,
                            edge_hidden=128,
                            gru_out=128,
                            gru_layers=1,
                            linear_hidden=128)

    config = {RUN_PARAMETERS_KEY: run_parameters,
              TRAINING_PARAMETERS_KEY: training_parameters,
              MODEL_PARAMETERS_KEY: model_parameters}

    return config


@pytest.mark.parametrize("number_of_molecules", [20])
def test_smoke_test_experiment_driver(config, random_molecule_dataset, mpnn_model):

    _ = experiment_driver(config, random_molecule_dataset, mpnn_model)
