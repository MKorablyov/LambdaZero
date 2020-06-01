from typing import List

from torch_geometric.data import Data, InMemoryDataset


class FakeMoleculeDataset(InMemoryDataset):
    def __init__(self, list_molecules: List[Data]):
        super(FakeMoleculeDataset, self).__init__(None,
                                                  transform=None,
                                                  pre_transform=None)
        self.data, self.slices = self.collate(list_molecules)

