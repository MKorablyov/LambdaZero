import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from LambdaZero.loggers.mlflow_logger import MLFlowLogger
from LambdaZero.oracle_models.chemprop_model import (
    GeometricChempropNet,
)
from LambdaZero.representation_learning.dataloader_utils import (
    get_geometric_dataloaders,
)
from LambdaZero.representation_learning.datasets import (
    D4GeometricMoleculesDataset,
)
from LambdaZero.representation_learning.experiment_driver import experiment_driver
from LambdaZero.representation_learning.model_trainer import MoleculeModelTrainer
from LambdaZero.oracle_models.message_passing_model import (
    MessagePassingNet,
)
from LambdaZero.representation_learning.parameter_inputs import (
    CONFIG_KEY,
    NON_CONFIG_KEY,
    EXECUTION_FILENAME_KEY,
    PATHS_KEY,
    MODEL_PARAMETERS_KEY,
    TRAINING_PARAMETERS_KEY,
    RUN_PARAMETERS_KEY,
)


@pytest.fixture
def number_of_node_features(real_molecule_dataset):
    molecule_graph = real_molecule_dataset[0]
    node_features = molecule_graph.x
    number_features = node_features.shape[1]
    return number_features


@pytest.fixture
def number_of_edge_features(real_molecule_dataset):
    molecule_graph = real_molecule_dataset[0]
    edge_features = molecule_graph.edge_attr
    number_features = edge_features.shape[1]
    return number_features


@pytest.fixture
def expected_raw_files():
    return [
        "dock_blocks105_walk40_clust.feather",
        "dock_blocks105_walk40_2_clust.feather",
    ]


@pytest.fixture
def expected_processed_file():
    return "d4_processed_data.pt"


@pytest.fixture
def data_dir(expected_raw_files, smiles_and_scores_dataframe):

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        data_dir_path = Path(tmp_dir_str)

        logging.info("creating fake raw data files")
        index_splits = np.array_split(
            smiles_and_scores_dataframe.index, len(expected_raw_files)
        )
        for filename, indices in zip(expected_raw_files, index_splits):
            sub_df = smiles_and_scores_dataframe.loc[indices].reset_index(drop=True)
            file_path = data_dir_path.joinpath(filename)
            sub_df.to_feather(file_path)
        yield data_dir_path
    logging.info("deleting test folder")


@pytest.fixture
def work_dir(expected_raw_files, expected_processed_file, real_molecule_dataset):
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
def paths(data_dir, work_dir, output_dir):
    paths = dict(
        data_directory=str(data_dir),
        working_directory=str(work_dir),
        output_directory=str(output_dir),
        tracking_uri=str(output_dir.joinpath("mlruns")),
    )
    return paths


@pytest.fixture
def driver_inputs(model_name):
    inputs = None
    if model_name == "MPNN":
        inputs = dict(
            model_class=MessagePassingNet,
            dataset_class=D4GeometricMoleculesDataset,
            model_trainer_class=MoleculeModelTrainer,
            get_dataloaders=get_geometric_dataloaders,
        )
    elif model_name == "geometric-chemprop":
        inputs = dict(
            model_class=GeometricChempropNet,
            dataset_class=D4GeometricMoleculesDataset,
            model_trainer_class=MoleculeModelTrainer,
            get_dataloaders=get_geometric_dataloaders,
        )

    return inputs


@pytest.fixture
def model_parameters(model_name, number_of_node_features, number_of_edge_features):
    parameters = None
    if model_name == "MPNN":
        parameters = dict(
            name="MPNN",
            node_feat=number_of_node_features,
            edge_feat=number_of_edge_features,
            gcn_size=8,
            edge_hidden=8,
            linear_hidden=8,
        )

    elif model_name == "geometric-chemprop":
        parameters = dict(name=model_name, depth=2, ffn_num_layers=2, ffn_hidden_size=8)
    return parameters


@pytest.fixture
def input_and_run_config(paths, model_parameters):

    run_parameters = dict(experiment_name="TEST", run_name="exp-driver-smoke-test")

    training_parameters = dict(
        num_epochs=10,
        num_workers=0,
        batch_size=2,
        learning_rate=1e-3,
        train_fraction=0.8,
        validation_fraction=0.1,
        patience=1,
    )

    config = {
        RUN_PARAMETERS_KEY: run_parameters,
        TRAINING_PARAMETERS_KEY: training_parameters,
        MODEL_PARAMETERS_KEY: model_parameters,
    }

    non_config = {PATHS_KEY: paths, EXECUTION_FILENAME_KEY: "test_file_name.py"}

    config_and_augmented = {CONFIG_KEY: config, NON_CONFIG_KEY: non_config}

    return config_and_augmented


@pytest.mark.parametrize("model_name", ["geometric-chemprop", "MPNN"])
def test_smoke_test_experiment_driver(input_and_run_config, driver_inputs):
    logger_class = MLFlowLogger
    with pytest.warns(None) as record:
        _ = experiment_driver(
            input_and_run_config, logger_class=logger_class, **driver_inputs
        )

    for warning in record.list:
        assert warning.category != UserWarning, "A user warning was raised"
