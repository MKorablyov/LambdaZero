import numpy as np


def get_geometric_center(list_positions: np.array) -> np.array:
    assert (
        list_positions.shape[1] == 3
    ), "error: the rows are expected to be 3D vectors."
    return list_positions.mean(axis=0)


def get_positions_relative_to_center(list_positions: np.array) -> np.array:
    center = get_geometric_center(list_positions)
    list_relative_positions = list_positions - center
    return list_relative_positions


def get_quadratic_position_tensor(list_positions: np.array) -> np.array:

    # create the N x 3 array of relative positions
    relative_positions = get_positions_relative_to_center(list_positions)

    t = np.dot(relative_positions.T, relative_positions) / relative_positions.shape[0]

    return t


def diagonalize_quadratic_position_tensor(quadratic_position_tensor: np.array):
    eigenvalues, raw_u_matrix = np.linalg.eigh(quadratic_position_tensor)
    return eigenvalues, raw_u_matrix


def get_direction(x2: np.float, tol=1e-8) -> np.float:
    if np.abs(x2) < tol:
        return 1.0
    else:
        return np.sign(x2)


def get_quadratic_position_tensor_basis(
    list_relative_positions: np.array, raw_u_matrix: np.array
) -> np.array:
    """
    The quadratic position tensor T^{alpha,beta} can be used to define the principal axes of the
    molecule. However, there remains arbitrariness in the direction of the eigenvectors; +v or -v
    will work equally well. This function removes this arbitrariness by choosing eigenvectors that point
    in the direction of maximum molecular weight.

    Args:
        list_relative_positions (np.array): positions of the atoms relative to its center
        raw_u_matrix (np.array): matrix containing eigenvectors of the quadratic position tensor

    Returns:
        u_matrix (np.array): eigenvectors with arbitrariness lifted.
    """

    projected_positions = np.dot(list_relative_positions, raw_u_matrix)

    signed_projected_positions_squared = (
        np.sign(projected_positions) * projected_positions ** 2
    )

    list_summed_projected_positions_squared = signed_projected_positions_squared.sum(
        axis=0
    )

    v1 = get_direction(list_summed_projected_positions_squared[0]) * raw_u_matrix[:, 0]
    v2 = get_direction(list_summed_projected_positions_squared[1]) * raw_u_matrix[:, 1]
    v3 = np.cross(v1, v2)

    normalized_u_matrix = np.stack([v1, v2, v3], axis=1)
    return normalized_u_matrix


def get_normalized_u_matrix(list_positions: np.array) -> np.array:
    """
    This method computes the quadratic position tensor and fixes the
    orientation arbitrariness.
    """

    quadratic_position_tensor = get_quadratic_position_tensor(list_positions)

    _, raw_u_matrix = diagonalize_quadratic_position_tensor(quadratic_position_tensor)

    relative_positions = get_positions_relative_to_center(list_positions)

    u_matrix = get_quadratic_position_tensor_basis(relative_positions, raw_u_matrix)

    return u_matrix
