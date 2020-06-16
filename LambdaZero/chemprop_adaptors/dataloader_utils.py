import logging
from typing import Dict, Union

from torch.utils.data import Dataset, DataLoader

from LambdaZero.datasets.dataset_splitting import DictKnnDatasetSplitter
from LambdaZero.examples.chemprop.ChempropRegressor import chemprop_collate_fn


def get_chemprop_dataloaders(dataset: Dataset,
                             training_parameters: Dict[str, Union[int, float]],
                             random_seed: int):

    splitter = DictKnnDatasetSplitter(train_fraction=training_parameters["train_fraction"],
                                      validation_fraction=training_parameters["validation_fraction"],
                                      random_seed=random_seed)

    training_dataset, validation_dataset, test_dataset = splitter.get_split_datasets(dataset)

    logging.info(f"size of training set {len(training_dataset)}")
    logging.info(f"size of validation set {len(validation_dataset)}")
    logging.info(f"size of test set {len(test_dataset)}")

    training_dataloader = DataLoader(training_dataset,
                                     batch_size=training_parameters["batch_size"],
                                     num_workers=training_parameters['num_workers'],
                                     collate_fn=chemprop_collate_fn,
                                     shuffle=True)

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=training_parameters["batch_size"],
                                       num_workers=training_parameters['num_workers'],
                                       collate_fn=chemprop_collate_fn,
                                       shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=training_parameters['batch_size'],
                                 num_workers=training_parameters['num_workers'],
                                 collate_fn=chemprop_collate_fn,
                                 shuffle=False)

    return training_dataloader, validation_dataloader, test_dataloader


