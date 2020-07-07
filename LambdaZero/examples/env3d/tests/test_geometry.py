import pytest
import numpy as np
import scipy.spatial

from LambdaZero.examples.env3d.geometry import (
    get_geometric_center,
    get_positions_relative_to_center,
    get_quadratic_position_tensor,
    diagonalize_quadratic_position_tensor,
)


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


@pytest.fixture
def expected_relative_positions(positions, expected_center):

    list_relative_positions = []
    for p in positions:
        list_relative_positions.append(p - expected_center)

    return np.array(list_relative_positions)


@pytest.fixture
def expected_tensor(expected_relative_positions):
    number_of_positions = len(expected_relative_positions)

    t = np.zeros([3, 3])

    for p in expected_relative_positions:
        for alpha in range(3):
            for beta in range(3):
                t[alpha, beta] += p[alpha] * p[beta]

    t /= number_of_positions
    return t


@pytest.fixture
def random_eigenvalues():
    np.random.seed(231412)
    return np.sort(np.random.rand(3))


@pytest.fixture
def random_rotation():
    np.random.seed(231412)

    vector = np.random.rand(3)
    vector = vector / np.linalg.norm(vector)
    angle = 2 * np.pi * np.random.rand()
    rotation_matrix = scipy.spatial.transform.Rotation.from_rotvec(
        angle * vector
    ).as_matrix()
    return rotation_matrix


@pytest.fixture
def random_quadratic_tensor(random_eigenvalues, random_rotation):
    lbda = np.diag(random_eigenvalues)
    t = np.dot(np.dot(random_rotation, lbda), random_rotation.T)
    return t


def test_get_geometric_center(positions, expected_center):
    computed_center = get_geometric_center(positions)

    np.testing.assert_almost_equal(computed_center, expected_center)


def test_get_geometric_center_assert():
    bad_positions = np.random.rand(3, 8)
    with pytest.raises(AssertionError):
        _ = get_geometric_center(bad_positions)


def test_get_positions_relative_to_center(positions, expected_relative_positions):
    computed_relative_positions = get_positions_relative_to_center(positions)

    np.testing.assert_almost_equal(
        computed_relative_positions, expected_relative_positions
    )


def test_get_quadratic_position_tensor(positions, expected_tensor):

    computed_tensor = get_quadratic_position_tensor(positions)
    np.testing.assert_almost_equal(computed_tensor, expected_tensor)


def test_diagonalize_quadratic_position_tensor(
    random_eigenvalues, random_rotation, random_quadratic_tensor
):
    computed_eigenvalues, computed_u_matrix = diagonalize_quadratic_position_tensor(
        random_quadratic_tensor
    )

    np.testing.assert_almost_equal(computed_eigenvalues, random_eigenvalues)
    np.testing.assert_almost_equal(computed_u_matrix, random_rotation)
