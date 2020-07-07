import numpy as np


def get_geometric_center(list_positions: np.array) -> np.array:
    assert list_positions.shape[1] == 3, "error: the rows are expected to be 3D vectors."
    return list_positions.mean(axis=0)
