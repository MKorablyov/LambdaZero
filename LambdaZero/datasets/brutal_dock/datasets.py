import logging
import shutil
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset

from LambdaZero.datasets.brutal_dock.dataset_utils import get_smiles_and_scores_from_feather, get_molecule_graphs_from_smiles_and_scores


class D4MoleculesDataset(InMemoryDataset):
    feather_filename = 'dock_blocks105_walk40_clust.feather'

    def __init__(self, root_dir: str, original_raw_data_dir: str):
        self.original_raw_data_dir_path = Path(original_raw_data_dir)

        super(D4MoleculesDataset, self).__init__(root_dir, transform=None, pre_transform=None)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.feather_filename]

    @property
    def processed_file_names(self):
        return ['d4_processed_data.pt']

    # TODO: This download method could be in a base class
    def download(self):
        # Download to `self.raw_dir`.
        raw_dir_path = Path(self.raw_dir)

        for raw_file_name in self.raw_file_names:
            raw_data_path = raw_dir_path.joinpath(raw_file_name)
            original_data_path = self.original_raw_data_dir_path.joinpath(raw_file_name)

            if not raw_data_path.is_file():
                logging.info(f"Copying {original_data_path} to {raw_data_path})")
                shutil.copy(str(original_data_path), str(raw_data_path))

    def process(self):
        # Read data into huge `Data` list.
        logging.info("Processing the raw data from the Feather file to a Data object saved on disk")

        feather_data_path = Path(self.raw_dir).joinpath(self.raw_file_names[0])
        list_smiles, list_scores = get_smiles_and_scores_from_feather(feather_data_path)
        data_list = get_molecule_graphs_from_smiles_and_scores(list_smiles, list_scores)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
