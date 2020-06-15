import logging
import numpy as np
import tempfile
from pathlib import Path

import pytest
import ray
from ray.tune import tune

from LambdaZero.examples.chemprop.ChempropRegressor import ChempropRegressor


@pytest.fixture
def expected_raw_files():
    return [
        "dock_blocks105_walk40_clust.feather",
        "dock_blocks105_walk40_2_clust.feather",
    ]


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
def summaries_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str)
    logging.info("deleting test folder")


@pytest.fixture
def model_parameters():

    model_parameters = {
        "name": "chemprop",
        "bias": False,
        "hidden_size": 5,
        "depth": 2,
        "dropout": 0.0,
        "atom_messages": False,
        "undirected": False,
        "ffn_hidden_size": 7,
        "ffn_num_layers": 3,
    }

    return model_parameters


@pytest.fixture
def config(model_parameters, data_dir, summaries_dir):

    config = {
        "dataset_root": str(data_dir),
        "random_seed": 0,
        "target": "gridscore",
        "target_norm": [-26.3, 12.3],
        "lr": 0.001,
        "b_size": 64,
        "num_epochs": 1,
        "model_parameters": model_parameters,
        "summaries_dir": summaries_dir,
    }

    return config


def test_chemprop_regressor(config):
    tunable_regressor = ChempropRegressor(config=config)


def test_smoke_test_tuning_chemprop_regressor(config):
    """
    This is a SMOKE TEST. It just validates that the code will run without errors if given
    expected inputs. It does not validate that the results are correct.
    """
    ray.init(local_mode=True)

    _ = tune.run(
        ChempropRegressor,
        config=config,
        stop={"training_iteration": 2},
        resources_per_trial={"cpu": 1, "gpu": 0.0},
        num_samples=1,
        local_dir=config["summaries_dir"],
    )
