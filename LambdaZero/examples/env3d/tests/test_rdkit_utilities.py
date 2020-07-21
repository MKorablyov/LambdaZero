import pytest
import numpy as np

from LambdaZero.examples.env3d.rdkit_utilities import (
    find_index_of_lowest_converged_energy,
)


@pytest.fixture
def energy_converged_tuples():

    list_energies = [1., 3., 3., 2., 1.]
    list_converged = [1, 0, 1, 0, 1]

    list_tuples = [(c, e) for c, e in zip(list_converged, list_energies)]

    return list_tuples


@pytest.fixture
def expected_lowest_energy_index(energy_converged_tuples):
    converged_index = None
    lowest_energy = np.inf
    for index, (converged, energy) in enumerate(energy_converged_tuples):
        if converged == 0 and energy < lowest_energy:
            converged_index = index
            lowest_energy = energy

    return converged_index


def test_find_index_of_lowest_converged_energy(
    energy_converged_tuples, expected_lowest_energy_index
):
    computed_lowest_energy_index = find_index_of_lowest_converged_energy(
        energy_converged_tuples
    )
    assert computed_lowest_energy_index == expected_lowest_energy_index
