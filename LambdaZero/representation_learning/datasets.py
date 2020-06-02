import logging
import shutil
from pathlib import Path
from typing import List

import pandas as pd

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset

from LambdaZero.datasets.brutal_dock.chemprop_adaptors.utils import get_chemprop_graphs_from_raw_data_dataframe
from LambdaZero.representation_learning.dataset_utils import get_molecule_graphs_from_raw_data_dataframe


class MoleculesDatasetBase(Dataset):
    def __init__(self, root_dir: str, original_raw_data_dir: str):
        self.original_raw_data_dir_path = Path(original_raw_data_dir)
        self.root_dir_path = Path(root_dir)

    @classmethod
    def create_dataset(cls, root_dir: str, original_raw_data_dir: str):
        return cls(root_dir, original_raw_data_dir)

    def _move_files(self, list_raw_file_names: List[str], destination_directory: str):

        destination_directory_path = Path(destination_directory)

        for raw_file_name in list_raw_file_names:
            destination_path = destination_directory_path.joinpath(raw_file_name)
            origin_path = self.original_raw_data_dir_path.joinpath(raw_file_name)

            if not destination_path.is_file():
                logging.info(f"Copying {origin_path} to {destination_path})")
                shutil.copy(str(origin_path), str(destination_path))


class D4ChempropMoleculesDataset(MoleculesDatasetBase):
    raw_file_names = ["dock_blocks105_walk40_clust.feather", "dock_blocks105_walk40_2_clust.feather"]

    def __init__(self, root_dir: str, original_raw_data_dir: str):
        super(D4ChempropMoleculesDataset, self).__init__(root_dir, original_raw_data_dir)

        self._move_files(list_raw_file_names=self.raw_file_names,
                         destination_directory=root_dir)

        self._list_data = self._process()

    def _process(self):
        # Read data into huge `Data` list.
        logging.info("Processing the raw data from the Feather file to a chemprop objects")

        list_df = []
        for raw_file_name in self.raw_file_names:
            feather_data_path = self.root_dir_path.joinpath(raw_file_name)
            df = pd.read_feather(feather_data_path)
            list_df.append(df)

        raw_data_df = pd.concat(list_df).reset_index(drop=True)
        list_data = get_chemprop_graphs_from_raw_data_dataframe(raw_data_df)
        return list_data

    def __getitem__(self, index):
        return self._list_data[index]

    def __len__(self):
        return len(self._list_data)


class GeometricMoleculesDatasetBase(MoleculesDatasetBase, InMemoryDataset):
    def __init__(self, root_dir: str, original_raw_data_dir: str, transform=None, pre_transform=None):
        MoleculesDatasetBase.__init__(self, root_dir, original_raw_data_dir)
        InMemoryDataset.__init__(self, root_dir, transform=transform, pre_transform=pre_transform)


class D4GeometricMoleculesDataset(GeometricMoleculesDatasetBase):
    feather_filenames = ["dock_blocks105_walk40_clust.feather", "dock_blocks105_walk40_2_clust.feather"]

    def __init__(self, root_dir: str, original_raw_data_dir: str):
        super(D4GeometricMoleculesDataset, self).__init__(root_dir, original_raw_data_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.feather_filenames

    @property
    def processed_file_names(self):
        return ['d4_processed_data.pt']

    def download(self):
        self._move_files(list_raw_file_names=self.raw_file_names, destination_directory=self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        logging.info("Processing the raw data from the Feather file to a Data object saved on disk")

        list_df = []
        for raw_file_name in self.raw_file_names:
            feather_data_path = Path(self.raw_dir).joinpath(raw_file_name)
            df = pd.read_feather(feather_data_path)
            list_df.append(df)

        raw_data_df = pd.concat(list_df).reset_index(drop=True)
        data_list = get_molecule_graphs_from_raw_data_dataframe(raw_data_df)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
