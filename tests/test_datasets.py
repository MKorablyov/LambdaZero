import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LambdaZero.datasets.temp_brunos_work.datasets import D4MoleculesDataset


@pytest.fixture
def fake_raw_molecule_data_dataframe(realistic_smiles):
    np.random.seed(12312)
    size = len(realistic_smiles)
    list_gridscores = np.random.random(size)
    list_klabels = np.random.choice(range(10), size)

    df = pd.DataFrame(data={'smiles': realistic_smiles, 'gridscore': list_gridscores, 'klabel': list_klabels})

    return df


@pytest.fixture
def original_raw_data_dir(fake_raw_molecule_data_dataframe):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        raw_dir = Path(tmp_dir_str).joinpath("raw")
        raw_dir.mkdir()

        number_of_files = len(D4MoleculesDataset.feather_filenames)
        list_index_groups = np.array_split(fake_raw_molecule_data_dataframe.index, number_of_files)

        for filename, indices in zip(D4MoleculesDataset.feather_filenames, list_index_groups):
            file_path = raw_dir.joinpath(filename)
            sub_df = fake_raw_molecule_data_dataframe.iloc[indices].reset_index(drop=True)
            sub_df.to_feather(file_path)
        yield str(raw_dir)

    logging.info("deleting test folder")


@pytest.fixture
def root_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        yield tmp_dir_str

    logging.info("deleting test folder")


def test_d4_molecules_dataset(root_dir, original_raw_data_dir, fake_raw_molecule_data_dataframe):

    dataset = D4MoleculesDataset.create_dataset(root_dir, original_raw_data_dir)

    assert dataset.processed_dir == str(Path(root_dir).joinpath('processed-done/'))

    assert len(fake_raw_molecule_data_dataframe) == len(dataset)

    for column_name in ["gridscore", "klabel"]:
        expected_values = fake_raw_molecule_data_dataframe[column_name].values
        computed_values = [d[column_name].item() for d in dataset]
        np.testing.assert_almost_equal(expected_values, computed_values)
