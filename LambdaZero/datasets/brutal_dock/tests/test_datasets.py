import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset


@pytest.fixture
def fake_smiles_and_gridscore_dataframe():

    list_smiles = ['CC(C)=CC(C)(C)O', 'CC(C)(O)C(F)(F)F', 'O=[SH](=O)S(=O)(=O)O']
    list_scores = [1., 2., 3.]

    df = pd.DataFrame(data={'smiles': list_smiles, 'gridscore': list_scores})

    return df


@pytest.fixture
def original_raw_data_dir(fake_smiles_and_gridscore_dataframe):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        raw_dir = Path(tmp_dir_str).joinpath("raw")
        raw_dir.mkdir()
        file_path = raw_dir.joinpath(D4MoleculesDataset.feather_filename)
        fake_smiles_and_gridscore_dataframe.to_feather(file_path)
        yield str(raw_dir)

    logging.info("deleting test folder")


@pytest.fixture
def root_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        yield tmp_dir_str

    logging.info("deleting test folder")


def test_d4_molecules_dataset(root_dir, original_raw_data_dir, fake_smiles_and_gridscore_dataframe):

    dataset = D4MoleculesDataset(root_dir, original_raw_data_dir)

    assert dataset.processed_dir == str(Path(root_dir).joinpath('processed/'))

    assert len(fake_smiles_and_gridscore_dataframe) == len(dataset)

    expected_values = fake_smiles_and_gridscore_dataframe['gridscore'].values
    computed_values = [d.dockscore.item() for d in dataset]
    np.testing.assert_array_equal(expected_values, computed_values)
