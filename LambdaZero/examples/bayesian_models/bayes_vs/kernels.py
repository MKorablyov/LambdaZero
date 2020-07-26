"""
various functions to help compute kernel matrices between datapoints and the MMD between distributions.

"""

import warnings
import torch


def similarity_matrix_invesrse_quadratics(x, x2, c):
    sq_dist = square_distance_matrix_between_tensors(x, x2)
    res = c / (c + sq_dist)
    return res


def estimate_mmd_invesrse_quadratics(x1, x2):
    c = 2 * x1.shape[1] * (1 ** 2)
    assert x1.shape[0] == x2.shape[0]
    if x1.shape[0] == 1:
        warnings.warn("Computing MMD with only 1 example! (are you sure you meant to do this?)")
        return -2 * similarity_matrix_invesrse_quadratics(x1, x2, c).mean()
    else:
        x1_term = off_diagional_similarity_matrix_mean(lambda x: similarity_matrix_invesrse_quadratics(x, x, c), x1)
        x2_term = off_diagional_similarity_matrix_mean(lambda x: similarity_matrix_invesrse_quadratics(x, x, c), x2)
        x1_x2_terms = similarity_matrix_invesrse_quadratics(x1, x2, c).mean()
        return x1_term + x2_term - 2 * x1_x2_terms


def square_distance_matrix_between_tensors(x1, x2):
    """
    :param x1: [b_1, ...]
    :param x2: [b_2, ...]
    :return: [b_1, b_2] of the squared
    """
    x1_flattened = torch.flatten(x1, start_dim=1)
    x2_flattened = torch.flatten(x2, start_dim=1)
    x1_sq = torch.sum(x1_flattened**2, dim=1)[:, None]
    x2_sq = torch.sum(x2_flattened**2, dim=1)[None, :]
    x1x2 = x1_flattened @ x2_flattened.transpose(0, 1)
    sq_dist_mat = -2 * x1x2 + x2_sq + x1_sq
    return sq_dist_mat


def off_diagional_similarity_matrix_mean(sim_matrix_calculator, x):
    num_off_diagonal_terms = x.shape[0] * (x.shape[0] - 1)
    off_diagonal_sum = 2 * sim_matrix_calculator(x).triu(diagonal=1).sum()
    off_diagonal_mean = off_diagonal_sum / num_off_diagonal_terms
    return off_diagonal_mean





