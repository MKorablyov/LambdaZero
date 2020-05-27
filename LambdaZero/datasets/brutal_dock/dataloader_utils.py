import logging
from typing import Dict, Union

from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.data import Dataset as GeometricDataset

from LambdaZero.datasets.brutal_dock.dataset_splitting import KnnDatasetSplitter


def get_geometric_dataloaders(dataset: GeometricDataset,
                              training_parameters: Dict[str, Union[int, float]],
                              random_seed: int):

    splitter = KnnDatasetSplitter(training_parameters["train_fraction"],
                                  training_parameters["validation_fraction"],
                                  random_seed=random_seed)
    training_dataset, validation_dataset, test_dataset = splitter.get_split_datasets(dataset)

    logging.info(f"size of training set {len(training_dataset)}")
    logging.info(f"size of validation set {len(validation_dataset)}")
    logging.info(f"size of test set {len(test_dataset)}")

    training_dataloader = GeometricDataLoader(training_dataset,
                                              batch_size=training_parameters['batch_size'],
                                              num_workers=training_parameters['num_workers'],
                                              shuffle=True)
    validation_dataloader = GeometricDataLoader(validation_dataset,
                                                batch_size=training_parameters['batch_size'],
                                                num_workers=training_parameters['num_workers'],
                                                shuffle=True)
    test_dataloader = GeometricDataLoader(test_dataset,
                                          batch_size=training_parameters['batch_size'],
                                          num_workers=training_parameters['num_workers'],
                                          shuffle=False)

    return training_dataloader, validation_dataloader, test_dataloader


