import itertools

import pytest
import numpy as np
from torch_geometric.data import DataLoader

from LambdaZero.datasets.brutal_dock.dataset_splitting import _split_labels_in_groups, _get_group_indices, \
    get_split_datasets_by_knn


@pytest.fixture
def size():
    return 100

@pytest.fixture
def klabels(size):
    np.random.seed(23423)
    klabels = np.random.choice(list(range(10)), size)
    return klabels

@pytest.fixture
def probabilities():
    return [0.3, 0.4, 0.3]


def test_split_labels_in_groups(klabels, probabilities):

    list_groups = _split_labels_in_groups(klabels, probabilities)

    for klabel in klabels:
        number_of_times_found = 0
        for group in list_groups:
            if klabel in group:
                number_of_times_found += 1
        assert number_of_times_found == 1


def test_get_group_indices(klabels, probabilities):

    list_groups = _split_labels_in_groups(klabels, probabilities)
    list_group_indices = _get_group_indices(klabels, list_groups)

    for group_indices, group in zip(list_group_indices, list_groups):
        for index in group_indices:
            assert klabels[index] in group


@pytest.mark.parametrize("number_of_molecules", [100])
def test_get_split_datasets_by_knn(random_molecule_dataset):
    train_fraction = 0.5
    validation_fraction = 0.25
    training_dataset, validation_dataset, test_dataset = \
        get_split_datasets_by_knn(random_molecule_dataset, train_fraction, validation_fraction)

    set_of_training_klabels = set(data['klabel'].numpy()[0] for data in training_dataset[:])
    set_of_validation_klabels = set(data['klabel'].numpy()[0] for data in validation_dataset[:])
    set_of_test_klabels = set(data['klabel'].numpy()[0] for data in test_dataset[:])

    list_of_sets = [set_of_training_klabels, set_of_validation_klabels, set_of_test_klabels]

    for s1, s2 in itertools.combinations(list_of_sets, 2):
        assert len(s1.intersection(s2)) == 0

    for klabel in random_molecule_dataset.data.klabel.numpy():
        assert np.any(klabel in s for s in list_of_sets)
