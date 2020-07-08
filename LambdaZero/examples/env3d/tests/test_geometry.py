import itertools

import numpy as np
import pytest
import scipy.spatial

from LambdaZero.examples.env3d.geometry import get_center_of_mass, get_inertia_tensor


def get_epsilon():
    """
    The Levi-Civita fully anti-symmetric tensor
    """
    epsilon = np.zeros([3, 3, 3])
    for i, j, k in itertools.permutations([0, 1, 2], 3):
        m = np.zeros([3, 3])
        m[0, i] = 1
        m[1, j] = 1
        m[2, k] = 1
        sign = np.sign(np.linalg.det(m))
        epsilon[i, j, k] = sign

    return epsilon


def get_skew_symmetric_matrix(vector):
    epsilon = get_epsilon()
    matrix = np.einsum("ijk,j", epsilon, vector)
    return matrix


@pytest.fixture
def number_of_atoms():
    return 10


@pytest.fixture
def positions(number_of_atoms):
    np.random.seed(12312)
    return np.random.rand(number_of_atoms, 3)


@pytest.fixture
def masses(number_of_atoms):
    np.random.seed(34534)
    return 12 * np.random.rand(number_of_atoms)


@pytest.fixture
def expected_moment_of_inertia(masses, positions):
    inertia = np.zeros([3, 3])
    for mass, p in zip(masses, positions):
        matrix = get_skew_symmetric_matrix(p)
        inertia -= mass * np.dot(matrix, matrix)
    return inertia


@pytest.fixture
def expected_center_of_mass(masses, positions):
    center = np.zeros(3)

    total_mass = 0.0
    for m, p in zip(masses, positions):
        total_mass += m
        center += m * p
    center /= total_mass
    return center


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


def test_get_center_of_mass(masses, positions, expected_center_of_mass):
    computed_center = get_center_of_mass(masses, positions)

    np.testing.assert_almost_equal(computed_center, expected_center_of_mass)


@pytest.mark.parametrize("list_masses, list_positions", [(np.random.rand(3, 2), np.random.rand(3, 3)),
                                                         (np.random.rand(3), np.random.rand(3, 4, 5)),
                                                         (np.random.rand(3), np.random.rand(5, 3)),
                                                         (np.random.rand(3), np.random.rand(3, 2))])
def test_get_center_of_mass_assert(list_masses, list_positions):

    with pytest.raises(AssertionError):
        _ = get_center_of_mass(list_masses, list_positions)


def test_get_inertia_tensor(masses, positions, expected_moment_of_inertia):
    computed_moment_of_inertia = get_inertia_tensor(masses, positions)
    np.testing.assert_allclose(expected_moment_of_inertia, computed_moment_of_inertia)
