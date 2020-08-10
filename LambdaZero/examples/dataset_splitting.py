import logging
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from numpy.random.mtrand import RandomState
from torch.utils.data import Dataset, Subset


class AbstractDatasetSplitter:

    def __init__(self, train_fraction: float, validation_fraction: float, random_seed: int = 0):
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = 1. - train_fraction - validation_fraction

        assert 0. <= self.train_fraction < 1., "the training fraction should be between zero and one"
        assert 0. <= self.validation_fraction < 1., "the validation fraction should be between zero and one"
        assert 0. <= self.test_fraction < 1., "the implied test fraction should be between zero and one"

        self.random_generator = np.random.RandomState(seed=random_seed)
        self.random_generator_state = self.random_generator.get_state()

    def _get_dataset_sizes(self, full_dataset):
        dataset_size = len(full_dataset)
        train_size = int(self.train_fraction * dataset_size)
        valid_size = int(self.validation_fraction * dataset_size)
        test_size = dataset_size-train_size-valid_size

        return train_size, valid_size, test_size

    @abstractmethod
    def _get_train_validation_and_test_indices(self, random_generator: RandomState,
                                               full_dataset: Dataset,
                                               train_size: int, valid_size: int,
                                               test_size: int) -> Tuple[List[int], List[int], List[int]]:

        pass

    def get_split_datasets(self, full_dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        logging.info(f"Splitting data into train, validation, test sets")

        train_size, valid_size, test_size = self._get_dataset_sizes(full_dataset)

        # Initialize the random number generator with its original state, to insure
        # the same "random numbers" are produced every time, even if this method is called many times.
        self.random_generator.set_state(self.random_generator_state)

        train_indices, valid_indices, test_indices = \
                self._get_train_validation_and_test_indices(self.random_generator, full_dataset,
                                                           train_size, valid_size, test_size)

        training_dataset = Subset(full_dataset, train_indices)
        validation_dataset = Subset(full_dataset, valid_indices)
        test_dataset = Subset(full_dataset, test_indices)

        return training_dataset, validation_dataset, test_dataset


class RandomDatasetSplitter(AbstractDatasetSplitter):

    def _get_train_validation_and_test_indices(self, random_generator: RandomState,
                                               full_dataset: Dataset,
                                               train_size: int, valid_size: int,
                                               test_size: int) -> Tuple[List[int], List[int], List[int]]:

        list_indices = list(range(len(full_dataset)))

        random_generator.shuffle(list_indices)
        train_indices = list_indices[:train_size]
        valid_indices = list_indices[train_size: train_size+valid_size]
        test_indices = list_indices[train_size+valid_size:]
        assert len(test_indices) == test_size, "something is wrong with the size of the test set"
        return train_indices, valid_indices, test_indices


class KnnDatasetSplitter(AbstractDatasetSplitter):

    @classmethod
    def _split_labels_in_groups(cls, random_generator: RandomState,
                                list_klabels: List[int], list_probabilities: List[float]) -> List[List[int]]:
        """
        This method assigns each unique klabel to a group of labels. The group is chosen with
        probability given in list_probabilities.
        """
        assert np.isclose(np.sum(list_probabilities), 1.0), "probabilities do not sum up to 1"

        unique_labels = np.unique(list_klabels)

        group_numbers = np.arange(len(list_probabilities))

        list_groups = [[] for _ in group_numbers]

        for label in unique_labels:
            group_number = random_generator.choice(group_numbers, p=list_probabilities)
            list_groups[group_number].append(label)

        return list_groups

    @classmethod
    def _get_group_indices(cls, list_klabels: List[int], list_groups: List[List[int]]):
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

    def _get_train_validation_and_test_indices(self, random_generator: RandomState,
                                               full_dataset: Dataset,
                                               train_size: int, valid_size: int,
                                               test_size: int) -> Tuple[List[int], List[int], List[int]]:

        list_klabels = full_dataset.data["klabel"].numpy()

        total_size = len(full_dataset)

        list_probabilities = [train_size/total_size, valid_size/total_size, test_size/total_size]

        list_groups = self._split_labels_in_groups(random_generator, list_klabels, list_probabilities)
        train_indices, validation_indices, test_indices = self._get_group_indices(list_klabels, list_groups)

        random_generator.shuffle(train_indices)
        random_generator.shuffle(validation_indices)
        random_generator.shuffle(test_indices)

        return train_indices, validation_indices, test_indices

