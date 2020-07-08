import numpy as np


def get_center_of_mass(list_masses: np.array, list_positions: np.array) -> np.array:

    assert len(list_masses.shape) == 1, "error: list_masses should be a 1D array."

    assert len(list_positions.shape) == 2, "error: list_positions should be a 2D array."

    assert (
        list_masses.shape[0] == list_positions.shape[0]
    ), "error: there should be one mass for every position."

    assert (
        list_positions.shape[1] == 3
    ), "error: the rows are expected to be 3D vectors."

    list_mr = np.stack([list_masses, list_masses, list_masses], axis=1) * list_positions
    return list_mr.sum(axis=0) / np.sum(list_masses)


def get_inertia_contribution(mass: float, relative_position: np.array) -> np.array:
    """
    Contribution of a single (mass, position) to an inertia tensor.
    
    I^{alpha beta} = - m (r^alpha r^beta - |r|^2 delta^{alpha beta})
    """

    inertia_contribution = (
        mass * np.dot(relative_position, relative_position) * np.eye(3)
    )
    inertia_contribution -= mass * np.outer(relative_position, relative_position)
    return inertia_contribution


def get_inertia_tensor(list_masses, list_relative_positions) -> np.array:
    inertia_tensor = np.zeros([3, 3])
    for m, p in zip(list_masses, list_relative_positions):
        inertia_tensor += get_inertia_contribution(m, p)
    return inertia_tensor


def project_direction_out_of_tensor(tensor: np.array, direction: np.array) -> np.array:
    assert np.isclose(
        np.linalg.norm(direction), 1.0
    ), "direction should be a unit vector"

    v1 = np.dot(direction, tensor)
    v2 = np.dot(tensor, direction)
    o = np.dot(direction, v2)

    projected_tensor = (
        tensor
        - np.outer(direction, v1)
        - np.outer(v2, direction)
        + o * np.outer(direction, direction)
    )
    return projected_tensor


def diagonalize_quadratic_position_tensor(quadratic_position_tensor: np.array):
    eigenvalues, raw_u_matrix = np.linalg.eigh(quadratic_position_tensor)
    return eigenvalues, raw_u_matrix
