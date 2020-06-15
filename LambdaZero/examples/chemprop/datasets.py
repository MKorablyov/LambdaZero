import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from LambdaZero.chemprop_adaptors.utils import get_chemprop_graphs_from_raw_data_dataframe


class D4ChempropMoleculesDataset(Dataset):
    raw_file_names = ["dock_blocks105_walk40_clust.feather", "dock_blocks105_walk40_2_clust.feather"]

    def __init__(self, raw_data_dir: str):
        self.raw_data_dir_path = Path(raw_data_dir)

        self._list_data = self._process()

    def _process(self):
        # Read data into huge `Data` list.
        logging.info("Processing the raw data from the Feather file to a chemprop objects")

        list_df = []
        for raw_file_name in self.raw_file_names:
            feather_data_path = self.raw_data_dir_path.joinpath(raw_file_name)
            df = pd.read_feather(feather_data_path)
            list_df.append(df)

        raw_data_df = pd.concat(list_df).reset_index(drop=True)
        list_data = get_chemprop_graphs_from_raw_data_dataframe(raw_data_df)
        return list_data

    def __getitem__(self, index):
        return self._list_data[index]

    def __len__(self):
        return len(self._list_data)
