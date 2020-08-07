import numpy as np


def get_angles_in_degrees(angles_in_radian):
    return angles_in_radian * 180.0 / np.pi


def get_angle_between_zero_and_two_pi(angle_in_radian: float):
    """
    For a given angle, return the equivalent angle between zero and 2 pi.

    Args:
        angle_in_radian (float): angle in radian

    Returns:
        angle: angle between zero and 2 pi.

    """
    return np.mod(angle_in_radian, 2 * np.pi)
