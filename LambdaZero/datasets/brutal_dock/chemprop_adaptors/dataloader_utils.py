import logging
from typing import Dict, Union, List

import torch
from chemprop.features import BatchMolGraph
from torch.utils.data import Dataset, DataLoader

from LambdaZero.datasets.brutal_dock.dataset_splitting import DictKnnDatasetSplitter


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


def chemprop_collate_fn(list_dict: List[Dict]):

        collated_dict = dict()
        for key in list_dict[0].keys():
            if key == 'mol_graph':
                values = BatchMolGraph([d['mol_graph'] for d in list_dict])
            else:
                values = torch.tensor([d[key] for d in list_dict])
            collated_dict[key] = values

        return collated_dict