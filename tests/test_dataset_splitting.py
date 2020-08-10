import itertools

import numpy as np
import pytest

from LambdaZero.examples.dataset_splitting import KnnDatasetSplitter, RandomDatasetSplitter


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


@pytest.mark.parametrize("number_of_molecules, splitter_class", ([100, RandomDatasetSplitter],
                                                                 [100, KnnDatasetSplitter]))
def test_dataset_splitter_accessible_subset(splitter_class, random_molecule_dataset, probabilities):
    train_fraction = probabilities[0]
    validation_fraction = probabilities[1]
    random_seed = 123

    splitter = splitter_class(train_fraction, validation_fraction, random_seed)
    training_dataset, validation_dataset, test_dataset = splitter.get_split_datasets(random_molecule_dataset)

    # Pytorch is very picky about what can serve as subset indices; test that we can access elements one by one
    _ = training_dataset[0]
    _ = validation_dataset[0]
    _ = test_dataset[0]


@pytest.mark.parametrize("number_of_molecules, splitter_class", ([100, RandomDatasetSplitter],
                                                                 [100, KnnDatasetSplitter]))
def test_dataset_splitter_non_overlapping_sets(splitter_class, random_molecule_dataset, probabilities):
    train_fraction = probabilities[0]
    validation_fraction = probabilities[1]
    random_seed = 123

    splitter = splitter_class(train_fraction, validation_fraction, random_seed)
    training_dataset, validation_dataset, test_dataset = splitter.get_split_datasets(random_molecule_dataset)

    all_test_tags = set(random_molecule_dataset.data.test_tag.numpy())

    training_tags = set(d.test_tag.numpy()[0] for d in training_dataset)
    validation_tags = set(d.test_tag.numpy()[0] for d in validation_dataset)
    testing_tags = set(d.test_tag.numpy()[0] for d in test_dataset)

    list_of_sets = [training_tags, validation_tags, testing_tags]

    for tag in all_test_tags:
        assert tag in training_tags or tag in validation_tags or tag in testing_tags

    for split_set in list_of_sets:
        assert split_set.issubset(all_test_tags)

    for s1, s2 in itertools.combinations(list_of_sets, 2):
        assert len(s1.intersection(s2)) == 0


@pytest.mark.parametrize("number_of_molecules, splitter_class", ([100, RandomDatasetSplitter],
                                                                 [100, KnnDatasetSplitter]))
def test_dataset_splitter_idempotent_split(splitter_class, random_molecule_dataset, probabilities):
    train_fraction = probabilities[0]
    validation_fraction = probabilities[1]
    random_seed = 123

    splitter = splitter_class(train_fraction, validation_fraction, random_seed)
    training_dataset1, validation_dataset1, test_dataset1 = splitter.get_split_datasets(random_molecule_dataset)
    training_dataset2, validation_dataset2, test_dataset2 = splitter.get_split_datasets(random_molecule_dataset)

    list_pairs = [[training_dataset1, training_dataset2],
                  [validation_dataset1, validation_dataset2],
                  [test_dataset1, test_dataset2]]

    for dataset1, dataset2 in list_pairs:
        tags1 = set(d.test_tag.numpy()[0] for d in dataset1)
        tags2 = set(d.test_tag.numpy()[0] for d in dataset2)
        assert tags1 == tags2


def test_split_labels_in_groups(klabels, probabilities):
    random_generator = np.random.RandomState(seed=123)
    list_groups = KnnDatasetSplitter._split_labels_in_groups(random_generator, klabels, probabilities)

    for klabel in klabels:
        number_of_times_found = 0
        for group in list_groups:
            if klabel in group:
                number_of_times_found += 1
        assert number_of_times_found == 1


def test_get_group_indices(klabels, probabilities):

    random_generator = np.random.RandomState(seed=123)
    list_groups = KnnDatasetSplitter._split_labels_in_groups(random_generator, klabels, probabilities)
    list_group_indices = KnnDatasetSplitter._get_group_indices(klabels, list_groups)

    for group_indices, group in zip(list_group_indices, list_groups):
        for index in group_indices:
            assert klabels[index] in group


@pytest.mark.parametrize("number_of_molecules", [100])
def test_get_split_datasets_by_knn(random_molecule_dataset):
    train_fraction = 0.5
    validation_fraction = 0.25
    splitter = KnnDatasetSplitter(train_fraction, validation_fraction, random_seed=123)

    training_dataset, validation_dataset, test_dataset = splitter.get_split_datasets(random_molecule_dataset)

    set_of_training_klabels = set(data['klabel'].numpy()[0] for data in training_dataset[:])
    set_of_validation_klabels = set(data['klabel'].numpy()[0] for data in validation_dataset[:])
    set_of_test_klabels = set(data['klabel'].numpy()[0] for data in test_dataset[:])

    list_of_sets = [set_of_training_klabels, set_of_validation_klabels, set_of_test_klabels]

    for s1, s2 in itertools.combinations(list_of_sets, 2):
        assert len(s1.intersection(s2)) == 0

    for klabel in random_molecule_dataset.data.klabel.numpy():
        assert np.any(klabel in s for s in list_of_sets)
