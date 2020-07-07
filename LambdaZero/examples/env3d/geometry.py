import numpy as np


def get_geometric_center(list_positions: np.array) -> np.array:
    return list_positions.mean(axis=0)
