import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from LambdaZero.examples.env3d.dataset.io_utilities import (
    append_row_to_dataframe,
    create_or_append_feather_file,
)


@pytest.fixture
def dataframe():
    np.random.seed(2342)
    data = {"a": np.random.rand(100), "b": np.random.rand(100)}
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def feather_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        feather_file_path = Path(tmp_dir).joinpath("test.feather")
        yield feather_file_path


def test_append_row_to_dataframe(dataframe):

    df = dataframe.iloc[:-1]
    last_row = dataframe.iloc[-1].to_dict()
    computed_df = append_row_to_dataframe(df, last_row)

    pd.testing.assert_frame_equal(computed_df, dataframe)


def test_create_or_append_feather_file(feather_path, dataframe):

    for index, row_series in dataframe.iterrows():
        row = row_series.to_dict()
        create_or_append_feather_file(feather_path, row)

    computed_df = pd.read_feather(feather_path)

    pd.testing.assert_frame_equal(computed_df, dataframe)
