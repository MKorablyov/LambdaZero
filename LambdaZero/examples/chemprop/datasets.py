import logging
import shutil
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import Dataset

from LambdaZero.chemprop_adaptors.utils import get_chemprop_graphs_from_raw_data_dataframe


class D4ChempropMoleculesDataset(Dataset):
    raw_file_names = ["dock_blocks105_walk40_clust.feather", "dock_blocks105_walk40_2_clust.feather"]

    def __init__(self, root_dir: str, original_raw_data_dir: str):
        self.original_raw_data_dir_path = Path(original_raw_data_dir)
        self.root_dir_path = Path(root_dir)

        self._move_files(list_raw_file_names=self.raw_file_names,
                         destination_directory=root_dir)

        self._list_data = self._process()

    def _move_files(self, list_raw_file_names: List[str], destination_directory: str):

        destination_directory_path = Path(destination_directory)

        for raw_file_name in list_raw_file_names:
            destination_path = destination_directory_path.joinpath(raw_file_name)
            origin_path = self.original_raw_data_dir_path.joinpath(raw_file_name)

            if not destination_path.is_file():
                logging.info(f"Copying {origin_path} to {destination_path})")
                shutil.copy(str(origin_path), str(destination_path))

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