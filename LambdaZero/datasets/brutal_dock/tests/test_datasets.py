from pathlib import Path
import numpy as np
from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset


def test_d4_molecules_dataset(root_dir, original_raw_data_dir, fake_raw_molecule_data_dataframe):

    dataset = D4MoleculesDataset.create_dataset(root_dir, original_raw_data_dir)

    assert dataset.processed_dir == str(Path(root_dir).joinpath('processed/'))

    assert len(fake_raw_molecule_data_dataframe) == len(dataset)

    for column_name in ["gridscore", "klabel"]:
        expected_values = fake_raw_molecule_data_dataframe[column_name].values
        computed_values = [d[column_name].item() for d in dataset]
        np.testing.assert_almost_equal(expected_values, computed_values)
