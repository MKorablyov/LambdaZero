import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from ray.tune import tune

from LambdaZero.examples.chemprop.ChempropRegressor import ChempropRegressor
from LambdaZero.loggers.wandb_logger import set_wandb_to_dryrun


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
        "experiment_name": f"TEST-{np.random.randint(1e3)}",
        "dataset_root": str(data_dir),
        "random_seed": 0,
        "target": "gridscore",
        "target_norm": [-26.3, 12.3],
        "lr": 0.001,
        "batch_size": 2,
        "model_parameters": model_parameters,
    }

    return config


@pytest.fixture
def mean():
    np.random.seed(12312)
    return torch.tensor(np.random.rand())


@pytest.fixture
def std():
    np.random.seed(22)
    return torch.tensor(np.random.rand())


@pytest.fixture
def random_values():
    return torch.rand(10)


def test_chemprop_regressor_normalizer(random_values, mean, std):

    computed_normalized_values = ChempropRegressor._normalize_target(
        random_values, mean, std
    )
    expected_normalized_values = (random_values - mean) / std

    np.testing.assert_allclose(
        computed_normalized_values.numpy(),
        expected_normalized_values.numpy(),
        atol=1e-6,
    )

    computed_values = ChempropRegressor._denormalize_target(
        expected_normalized_values, mean, std
    )

    np.testing.assert_allclose(
        computed_values.numpy(), random_values.numpy(), atol=1e-6
    )


def test_chemprop_regressor_get_size_of_batch(random_values, config):

    batch = {config["target"]: random_values, "other": "test"}

    expected_size = len(random_values)
    computed_size = ChempropRegressor._get_size_of_batch(batch, config)

    assert expected_size == computed_size


def test_get_logging_metrics(config, summaries_dir):

    std = config["target_norm"][1]
    average_epoch_loss = 32.56

    computed_results_dict = ChempropRegressor._get_logging_metrics(
        average_epoch_loss, config
    )

    rmse = std * np.sqrt(average_epoch_loss)

    expected_results_dict = {
        "mse_normalized_units": average_epoch_loss,
        "rmse_original_units": rmse,
    }

    assert computed_results_dict == expected_results_dict


def test_smoke_test_tuning_chemprop_regressor(config, summaries_dir):
    """
    This is a SMOKE TEST. It just validates that the code will run without errors if given
    expected inputs. It does not validate that the results are correct.
    """

    set_wandb_to_dryrun()
    ray.init()

    _ = tune.run(
        ChempropRegressor,
        config=config,
        stop={"training_iteration": 10},
        resources_per_trial={"cpu": 1, "gpu": 0.0},
        num_samples=1,
        local_dir=summaries_dir,
    )
