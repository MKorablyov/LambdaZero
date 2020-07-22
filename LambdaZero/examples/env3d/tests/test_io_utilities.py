import pytest
import numpy as np
import pandas as pd

from LambdaZero.examples.env3d.dataset.io_utilities import append_row_to_dataframe


@pytest.fixture
def dataframe():
    np.random.seed(2342)
    data = {"a": np.random.rand(100), "b": np.random.rand(100)}
    df = pd.DataFrame(data=data)
    return df


def test_append_row_to_dataframe(dataframe):

    df = dataframe.iloc[:-1]
    last_row = dataframe.iloc[-1].to_dict()
    computed_df = append_row_to_dataframe(df, last_row)

    pd.testing.assert_frame_equal(computed_df, dataframe)
