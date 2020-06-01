import logging
import shutil
from pathlib import Path
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset

from LambdaZero.representation_learning.dataset_utils import get_molecule_graphs_from_raw_data_dataframe


class MoleculesDatasetBase(InMemoryDataset):
    def __init__(self, root_dir: str, original_raw_data_dir: str, transform=None, pre_transform=None):
        self.original_raw_data_dir_path = Path(original_raw_data_dir)
        super(MoleculesDatasetBase, self).__init__(root_dir, transform=transform, pre_transform=pre_transform)

    @classmethod
    def create_dataset(cls, root_dir: str, original_raw_data_dir: str):
        return cls(root_dir, original_raw_data_dir)


class D4MoleculesDataset(MoleculesDatasetBase):
    feather_filenames = ["dock_blocks105_walk40_clust.feather", "dock_blocks105_walk40_2_clust.feather"]

    def __init__(self, root_dir: str, original_raw_data_dir: str):
        super(D4MoleculesDataset, self).__init__(root_dir, original_raw_data_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.feather_filenames

    @property
    def processed_file_names(self):
        return ['d4_processed_data.pt']

    # TODO: this should really be in the base class, but putting it there naively breaks the
    #  code because of pytorch_geometric's funny way of looking for the download method by introspection.
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
