import numpy as np


def get_geometric_center(list_positions: np.array) -> np.array:
    assert list_positions.shape[1] == 3, "error: the rows are expected to be 3D vectors."
    return list_positions.mean(axis=0)


def get_positions_relative_to_center(list_positions: np.array) -> np.array:
    center = get_geometric_center(list_positions)
    list_relative_positions = list_positions - center
    return list_relative_positions


def get_quadratic_position_tensor(list_positions: np.array) -> np.array:

    # create the N x 3 array of relative positions
    relative_positions = get_positions_relative_to_center(list_positions)

    t = np.dot(relative_positions.T, relative_positions)/relative_positions.shape[0]

    return t


def diagonalize_quadratic_position_tensor(quadratic_position_tensor: np.array):
    eigenvalues, u_matrix = np.linalg.eigh(quadratic_position_tensor)
    # ensure that the eigenvector matrix is a proper rotation
    u_matrix[:, 2] = np.cross(u_matrix[:, 0], u_matrix[:, 1])
    return eigenvalues, u_matrix


def get_relative_positions_in_quadratic_tensor_basis(list_positions: np.array) -> np.array:
    """
    This method computes the relative quadratic position tensor and expresses positions relative
    to their center in the natural basis of the tensor.
    """

    quadratic_position_tensor = get_quadratic_position_tensor(list_positions)

    _, u_matrix = diagonalize_quadratic_position_tensor(quadratic_position_tensor)

    relative_positions = get_positions_relative_to_center(list_positions)

    # the u_matrix contains the tensor's eigenvectors as columns
    relative_positions_in_tensor_basis = np.dot(relative_positions, u_matrix)

    return relative_positions_in_tensor_basis

