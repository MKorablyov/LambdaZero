import logging
import string
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset


@pytest.fixture
def fake_smiles_and_gridscore_dataframe():
    list_smiles = np.array(['CC(C)=CC(C)(C)O', 'CC(C)=CC(=O)NC(C#N)P(=O)(O)O',
                            'O=[SH](=O)S(=O)(=O)O', 'CC(C)=CN1CCN(P(=O)(O)O)CC1',
                            'CC(C)(O)C(F)(F)F', 'Nn1cnc2c([PH](=O)[O-])ncnc21',
                            'c1ccc2cc(N3CCOCC3)ccc2c1', 'CC(C)(O)Br', 'CC(=O)N[SH](=O)=O',
                            'CC(C)=CC1CC(C(NC(=O)C(C)O)NC(=O)S(=O)(=O)O)N(c2ccc3ccccc3c2)C1C(C)C'])

    size = len(list_smiles)
    list_scores = np.random.random(size)

    df = pd.DataFrame(data={'smiles': list_smiles, 'gridscore': list_scores})

    return df


@pytest.fixture
def original_raw_data_dir(fake_smiles_and_gridscore_dataframe):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        raw_dir = Path(tmp_dir_str).joinpath("raw")
        raw_dir.mkdir()

        number_of_files = len(D4MoleculesDataset.feather_filenames)
        list_index_groups = np.array_split(fake_smiles_and_gridscore_dataframe.index, number_of_files)

        for filename, indices in zip(D4MoleculesDataset.feather_filenames, list_index_groups) :
            file_path = raw_dir.joinpath(filename)
            sub_df = fake_smiles_and_gridscore_dataframe.iloc[indices].reset_index(drop=True)
            sub_df.to_feather(file_path)
        yield str(raw_dir)

    logging.info("deleting test folder")


@pytest.fixture
def root_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        yield tmp_dir_str

    logging.info("deleting test folder")


def test_d4_molecules_dataset(root_dir, original_raw_data_dir, fake_smiles_and_gridscore_dataframe):

    dataset = D4MoleculesDataset.create_dataset(root_dir, original_raw_data_dir)

    assert dataset.processed_dir == str(Path(root_dir).joinpath('processed/'))

    assert len(fake_smiles_and_gridscore_dataframe) == len(dataset)

    expected_values = fake_smiles_and_gridscore_dataframe['gridscore'].values
    computed_values = [d.dockscore.item() for d in dataset]
    np.testing.assert_almost_equal(expected_values, computed_values)
