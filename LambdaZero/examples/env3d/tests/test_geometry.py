import itertools

import numpy as np
import pytest
import scipy.spatial

from LambdaZero.examples.env3d.geometry import (
    get_center_of_mass,
    get_inertia_tensor,
    project_direction_out_of_tensor,
    multiply_scalars_and_vectors,
    rotate_single_point_about_axis,
    rotate_points_about_axis, get_positions_aligned_with_parent_inertia_tensor,
)


@pytest.fixture
def normalized_positions_and_masses():

    p1 = np.array([ 1., 0., 0.])
    p2 = np.array([-1., 0., 0.])

    p3 = np.array([ 0., 1., 0.])
    p4 = np.array([ 0.,-1., 0.])

    p5 = np.array([ 0., 0., 1.])
    p6 = np.array([ 0., 0.,-1.])


    positions = np.array([p1, p2, p3, p4, p5, p6])
    masses = np.array([10., 10., 5., 5., 1., 1.])

    return positions, masses



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
def random_axis_direction():
    np.random.seed(53242)
    vector = np.random.rand(3)
    n_axis = vector / np.linalg.norm(vector)
    return n_axis


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
def random_translation():
    np.random.seed(3442)
    return np.random.rand(3)

@pytest.fixture
def random_tensor(random_eigenvalues, random_rotation):
    lbda = np.diag(random_eigenvalues)
    t = np.dot(np.dot(random_rotation, lbda), random_rotation.T)
    return t


@pytest.fixture
def expected_projected_random_tensor(
    random_eigenvalues, random_rotation, random_axis_direction
):
    projected_random_tensor = np.zeros([3, 3])
    for eigenvalue, eigenvector in zip(random_eigenvalues, random_rotation.T):
        projected_eigenvector = eigenvector - random_axis_direction * np.dot(
            random_axis_direction, eigenvector
        )
        projected_random_tensor += eigenvalue * np.outer(
            projected_eigenvector, projected_eigenvector
        )
    return projected_random_tensor


@pytest.fixture
def number_of_parents_normalized_positions_and_masses():

    np.random.seed(34234)
    p1 = np.array([ 1., 0., 0.])
    p2 = np.array([-1., 0., 0.])

    p3 = np.array([ 0., 1., 0.])
    p4 = np.array([ 0.,-1., 0.])

    p5 = np.array([ 0., 0., 1.])
    p6 = np.array([ 0., 0.,-1.])

    p7 = np.random.rand(3)
    p8 = np.random.rand(3)
    p9 = np.random.rand(3)

    number_of_parent_atoms = 6
    positions = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9])
    masses = np.array([10., 10., 5., 5., 1., 1., 0.123, 0.5221, 1.553])

    return number_of_parent_atoms, positions, masses


def test_get_center_of_mass(masses, positions, expected_center_of_mass):
    computed_center = get_center_of_mass(masses, positions)

    np.testing.assert_almost_equal(computed_center, expected_center_of_mass)


def test_multiply_scalars_and_vectors():
    np.random.seed(6342)
    list_scalars = np.random.rand(10)
    list_vectors = np.random.rand(10, 3)

    expected_products = []
    for s, v in zip(list_scalars, list_vectors):
        expected_products.append(s * v)

    expected_products = np.array(expected_products)
    computed_products = multiply_scalars_and_vectors(list_scalars, list_vectors)
    np.testing.assert_allclose(expected_products, computed_products)


@pytest.mark.parametrize(
    "list_scalars, list_vectors",
    [
        (np.random.rand(3, 2), np.random.rand(3, 3)),
        (np.random.rand(3), np.random.rand(3, 4, 5)),
        (np.random.rand(3), np.random.rand(5, 3)),
        (np.random.rand(3), np.random.rand(3, 2)),
    ],
)
def test_multiply_scalars_and_vectors_assert(list_scalars, list_vectors):
    with pytest.raises(AssertionError):
        _ = multiply_scalars_and_vectors(list_scalars, list_vectors)


def test_get_inertia_tensor(masses, positions, expected_moment_of_inertia):
    computed_moment_of_inertia = get_inertia_tensor(masses, positions)
    np.testing.assert_allclose(expected_moment_of_inertia, computed_moment_of_inertia)


def test_project_direction_out_of_tensor(
    random_tensor, random_axis_direction, expected_projected_random_tensor
):
    computed_projected_random_tensor = project_direction_out_of_tensor(
        random_tensor, random_axis_direction
    )
    np.testing.assert_allclose(
        computed_projected_random_tensor, expected_projected_random_tensor
    )


def test_rotate_single_point_about_axis():
    fixed_point = np.array([0.0, 0.0, 0.0])
    n_axis = np.array([0.0, 0.0, 1.0])
    point = np.array([1.0, 0.0, 5.0])
    rotation_angle = np.pi / 2

    expected_rotated_point = np.array([0.0, 1.0, 5.0])
    computed_rotated_point = rotate_single_point_about_axis(
        fixed_point, n_axis, rotation_angle, point
    )

    np.testing.assert_almost_equal(expected_rotated_point, computed_rotated_point)


def test_rotate_points_about_axis(random_axis_direction, positions):
    np.random.seed(243523)
    rotation_angle = 2 * np.pi * np.random.rand()
    fixed_point = np.random.rand(3)

    computed_rotated_points = rotate_points_about_axis(
        positions, fixed_point, random_axis_direction, rotation_angle
    )

    expected_rotated_points = []
    for point in positions:
        p = rotate_single_point_about_axis(
            fixed_point, random_axis_direction, rotation_angle, point
        )
        expected_rotated_points.append(p)

    expected_rotated_points = np.array(expected_rotated_points)
    np.testing.assert_almost_equal(computed_rotated_points, expected_rotated_points)


def test_get_positions_aligned_with_parent_inertia_tensor(number_of_parents_normalized_positions_and_masses, random_rotation, random_translation):
    number_of_parent_atoms, normalized_positions, all_masses = number_of_parents_normalized_positions_and_masses

    all_positions = np.dot(normalized_positions, random_rotation) + random_translation

    computed_normalized_positions = get_positions_aligned_with_parent_inertia_tensor(all_positions, all_masses, number_of_parent_atoms)

    np.testing.assert_almost_equal(normalized_positions, computed_normalized_positions)
