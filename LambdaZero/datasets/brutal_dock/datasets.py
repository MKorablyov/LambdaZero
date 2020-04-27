from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset

from LambdaZero.datasets.brutal_dock.dataset_utils import get_smiles_and_scores_from_feather, get_molecule_graph_dataset


class D4MoleculesDataset(InMemoryDataset):
    feather_filename = 'dock_blocks105_walk40_clust.feather'

    def __init__(self, root_dir: str):
        super(D4MoleculesDataset, self).__init__(root_dir, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.feather_filename]

    @property
    def processed_file_names(self):
        return ['d4_processed_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        raise NotImplementedError("Raw data is not where it is expected: review code")

    def process(self):
        # Read data into huge `Data` list.

        feather_data_path = Path(self.raw_dir).joinpath(self.raw_file_names[0])
        list_smiles, list_scores = get_smiles_and_scores_from_feather(feather_data_path)
        data_list = get_molecule_graph_dataset(list_smiles, list_scores)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
