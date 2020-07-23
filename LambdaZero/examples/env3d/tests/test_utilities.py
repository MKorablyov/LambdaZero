from LambdaZero.examples.env3d.utilities import get_angles_in_degrees
import numpy as np


def test_get_angles_in_degrees():

    angles_in_radian = 0.5*np.pi
    expected_angles_in_degrees = 90.

    computed_angle_in_degrees = get_angles_in_degrees(angles_in_radian)

    np.testing.assert_almost_equal(expected_angles_in_degrees, computed_angle_in_degrees)
