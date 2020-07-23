import numpy as np
from scipy.spatial.transform import Rotation


def multiply_scalars_and_vectors(
    list_scalars: np.array, list_vectors: np.array
) -> np.array:
    """
    Multiply an array of scalar by an array of vector. For example:
    list_scalar = [1, 2, 3]
    list_vectors = [v1, v2, v3]
    then the product is list_new_vectors = [1*v1, 2*v2, 3*v3].

    Args:
        list_scalars (np.array): array of floats representing the scalars
        list_vectors (np.array): array of 3D vectors

    Returns:
        list_new_vectors (np.array): product of scalars and vectors

    """
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
    """
    Computes the center of mass of a set of points with masses and positions
    Args:
        list_masses (np.array): scalar masses
        list_positions (np.array): positions, as 3D vectors

    Returns:
        center_of_mass (np.array): a 3D vector
    """
    list_mr = multiply_scalars_and_vectors(list_masses, list_positions)
    return list_mr.sum(axis=0) / np.sum(list_masses)


def get_inertia_contribution(mass: np.float, relative_position: np.array) -> np.array:
    """
    Contribution of a single (mass, position) to an inertia tensor.

    I^{alpha beta} = - m (r^alpha r^beta - |r|^2 delta^{alpha beta})
    where "r" is the relative position, and "alpha, beta" are space coordinates.

    Args:
        mass (np.float): mass of a point
        relative_position (np.array): relative position of point

    Returns:
        inertia_tensor (np.array): inertia tensor, a 3 x 3 matrix
    """

    inertia_contribution = (
        mass * np.dot(relative_position, relative_position) * np.eye(3)
    )
    inertia_contribution -= mass * np.outer(relative_position, relative_position)
    return inertia_contribution


def get_inertia_tensor(list_masses, list_relative_positions) -> np.array:
    """
    Compute the full inertia tensor of a group of positions with corresponding masses.
    The inertia tensor is computed relative to the point to which the points are relative to.
    Args:
        list_masses (np.array): masses
        list_relative_positions (np.array): positions relative to a reference point

    Returns:
        inertia_tensor (np.array): inertia tensor, a 3 x 3 matrix
    """
    inertia_tensor = np.zeros([3, 3])
    for m, p in zip(list_masses, list_relative_positions):
        inertia_tensor += get_inertia_contribution(m, p)
    return inertia_tensor


def project_direction_out_of_tensor(tensor: np.array, direction: np.array) -> np.array:
    """
    This method projects out the direction from the tensor. That is to say that
        - direction . projected_tensor = 0
        - projected_tensor . tensor = 0

    Args:
        tensor (np.array): a 3x3 matrix
        direction (np.array): a unit 3D vector defining a direction

    Returns:
        projected_tensor (np.array): a 3x3 matrix

    """
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
    """

    Rotates a point around an axis pointing in the direction n_axis and going through fixed_point.

    Args:
        fixed_point (np.array): point on the rotation axis, which is held fixed by the rotation.
        n_axis (np.array): unit vector parallel to the axis of rotation, defining the direction of rotation
                           according to the right-hand rule.
        rotation_angle (np.float):  angle of rotation
        point (np.array): point to be rotated

    Returns:
        rotated_point (np.array): point after the rotation about the axis.

    """

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
    """

    Args:
        positions (np.array): points to be rotated
        fixed_point (np.array): point on the rotation axis, which is held fixed by the rotation.
        n_axis (np.array): unit vector parallel to the axis of rotation, defining the direction of rotation
                           according to the right-hand rule.
        rotation_angle (np.float):  angle of rotation

    Returns:
        rotated_points (np.array): points after the rotation about the axis.
    """

    rotation_matrix = Rotation.from_rotvec(rotation_angle * n_axis).as_matrix()

    center = positions.mean(axis=0)

    rotated_center = rotate_single_point_about_axis(
        fixed_point, n_axis, rotation_angle, center
    )

    relative_positions = positions - center
    rotated_relative_positions = np.dot(relative_positions, rotation_matrix.T)

    rotated_positions = rotated_center + rotated_relative_positions

    return rotated_positions


def get_positions_aligned_with_parent_inertia_tensor(
    all_positions: np.array, all_masses: np.array, number_of_parent_atoms: int
) -> np.array:

    parent_positions = all_positions[:number_of_parent_atoms]
    parent_masses = all_masses[:number_of_parent_atoms]

    parent_center_of_mass = get_center_of_mass(parent_masses, parent_positions)
    parent_inertia_cm = get_inertia_tensor(
        parent_masses, parent_positions - parent_center_of_mass
    )

    inertia_eigenvalues, u_matrix = np.linalg.eigh(parent_inertia_cm)

    normalized_positions = np.dot(all_positions - parent_center_of_mass, u_matrix)

    return normalized_positions
