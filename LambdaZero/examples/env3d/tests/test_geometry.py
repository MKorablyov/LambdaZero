import pytest
import numpy as np

from LambdaZero.examples.env3d.geometry import get_geometric_center


@pytest.fixture
def positions():
    np.random.seed(12312)
    return np.random.rand(10, 3)


@pytest.fixture
def expected_center(positions):
    center = np.zeros(3)

    number_of_positions = len(positions)

    for p in positions:
        center += p
    center /= number_of_positions
    return center


def test_get_geometric_center(positions, expected_center) -> np.array:
    computed_center = get_geometric_center(positions)

    np.testing.assert_almost_equal(computed_center, expected_center)

