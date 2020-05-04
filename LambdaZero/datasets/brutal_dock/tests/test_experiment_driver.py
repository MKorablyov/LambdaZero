import logging
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def best_model_path():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield Path(tmp_dir_str).joinpath("test_best_model_path")
    logging.info("deleting test folder")


@pytest.fixture
def input_config():
    config_dict = dict(num_epochs=100,
                       batch_size=4096,
                       lr=1e-4,
                       train_fraction=0.8,
                       validation_fraction=0.1,
                       num_workers=4)

