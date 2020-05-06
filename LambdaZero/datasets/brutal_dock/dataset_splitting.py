import logging

import torch
from torch.utils.data import Dataset


def get_split_datasets(full_dataset: Dataset, train_fraction: float, validation_fraction: float):
    dataset_size = len(full_dataset)
    train_size = int(train_fraction * dataset_size)
    valid_size = int(validation_fraction * dataset_size)
    test_size = dataset_size - train_size - valid_size

    logging.info(f"Splitting data into train, validation, test sets")
    training_dataset, validation_dataset, test_dataset = \
        torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size])

    return training_dataset, validation_dataset, test_dataset