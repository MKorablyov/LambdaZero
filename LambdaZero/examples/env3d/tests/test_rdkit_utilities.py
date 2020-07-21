import pytest
import numpy as np

from LambdaZero.examples.env3d.rdkit_utilities import (
    find_index_of_lowest_converged_energy,
)


list_clean_tuples = [(1., 1.), (0., 3.), (1, 3.), (0., 2.), (1., 1.)]
clean_index = 3

list_bad_tuples = [(1., 1.), (1., 2.)]
bad_index = np.NaN


@pytest.mark.parametrize("energy_converged_tuples, expected_lowest_energy_index",
                         [(list_clean_tuples, clean_index), (list_bad_tuples, bad_index)])
def test_find_index_of_lowest_converged_energy(
    energy_converged_tuples, expected_lowest_energy_index
):
    computed_lowest_energy_index = find_index_of_lowest_converged_energy(
        energy_converged_tuples
    )
    np.testing.assert_equal(computed_lowest_energy_index, expected_lowest_energy_index)
