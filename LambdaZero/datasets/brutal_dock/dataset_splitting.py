import logging
from typing import List

import torch
from torch.utils.data import Dataset, Subset
import numpy as np


def get_split_datasets(full_dataset: Dataset, train_fraction: float, validation_fraction: float):
    dataset_size = len(full_dataset)
    train_size = int(train_fraction * dataset_size)
    valid_size = int(validation_fraction * dataset_size)
    test_size = dataset_size - train_size - valid_size

    logging.info(f"Splitting data into train, validation, test sets")
    training_dataset, validation_dataset, test_dataset = \
        torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size])

    return training_dataset, validation_dataset, test_dataset


def _split_labels_in_groups(list_klabels: List[int], list_probabilities: List[float]) -> List[List[int]]:
    """
    This method assigns each unique klabel to a group of labels. The group is chosen with
    probability given in list_probabilities.
    """
    assert np.isclose(np.sum(list_probabilities), 1.0), "probabilities do not sum up to 1"

    unique_labels = np.unique(list_klabels)

    group_numbers = np.arange(len(list_probabilities))

    list_groups = [[] for _ in group_numbers]

    for label in unique_labels:
        group_number = np.random.choice(group_numbers, p=list_probabilities)
        list_groups[group_number].append(label)

    return list_groups


def _get_group_indices(list_klabels: List[int], list_groups: List[List[int]]):
    """
    This method assigns each klabel to a group, and returns the corresponding
    lists of indices.
    """
    all_indices = np.arange(len(list_klabels))
    list_group_indices = []

    for group in list_groups:
        group_mask = np.in1d(list_klabels, group)

        # pytorch really needs these to be vanilla integers, not np.int64
        group_indices = [int(i) for i in all_indices[group_mask]]
        list_group_indices.append(group_indices)

    return list_group_indices


def get_split_datasets_by_knn(full_dataset, train_fraction, validation_fraction):
    list_klabels = full_dataset.data["klabel"].numpy()

    test_fraction = 1. - train_fraction - validation_fraction
    list_probabilities = [train_fraction, validation_fraction, test_fraction]

    list_groups = _split_labels_in_groups(list_klabels, list_probabilities)
    list_group_indices = _get_group_indices(list_klabels, list_groups)

    split = []
    for group_indices in list_group_indices:
        np.random.shuffle(group_indices)
        randomized_indices = list(group_indices)

        subset = Subset(full_dataset, randomized_indices)
        split.append(subset)

    return split

