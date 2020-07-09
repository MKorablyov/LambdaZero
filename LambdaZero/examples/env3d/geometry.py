import numpy as np
from scipy.spatial.transform import Rotation


def multiply_scalars_and_vectors(
    list_scalars: np.array, list_vectors: np.array
) -> np.array:
    assert len(list_scalars.shape) == 1, "error: list_scalars should be a 1D array."

    assert len(list_vectors.shape) == 2, "error: list_vectors should be a 2D array."

    assert (
        list_scalars.shape[0] == list_vectors.shape[0]
    ), "error: there should be one scalar for every vector."

    assert (
        list_vectors.shape[1] == 3
    ), "error: the rows of the list of vectors are expected to be 3D vectors."

    return np.stack([list_scalars, list_scalars, list_scalars], axis=1) * list_vectors


def get_center_of_mass(list_masses: np.array, list_positions: np.array) -> np.array:
    list_mr = multiply_scalars_and_vectors(list_masses, list_positions)
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


def rotate_single_point_about_axis(
    fixed_point: np.array, n_axis: np.array, rotation_angle: np.float, point: np.array
):

    relative_position = point - fixed_point
    axis_projection = np.dot(relative_position, n_axis)

    x = relative_position - axis_projection * n_axis

    norm_x = np.linalg.norm(x)
    x_hat = x / norm_x
    y_hat = np.cross(n_axis, x_hat)

    rotated_x = norm_x * (
        np.cos(rotation_angle) * x_hat + np.sin(rotation_angle) * y_hat
    )

    rotated_position = fixed_point + axis_projection * n_axis + rotated_x
    return rotated_position


def rotate_points_about_axis(
    positions: np.array,
    fixed_point: np.array,
    n_axis: np.array,
    rotation_angle: np.float,
) -> np.array:

    rotation_matrix = Rotation.from_rotvec(rotation_angle * n_axis).as_matrix()

    center = positions.mean(axis=0)

    rotated_center = rotate_single_point_about_axis(
        fixed_point, n_axis, rotation_angle, center
    )

    relative_positions = positions - center
    rotated_relative_positions = np.dot(relative_positions, rotation_matrix.T)

    rotated_positions = rotated_center + rotated_relative_positions

    return rotated_positions
