import torch
from torch_geometric.data import InMemoryDataset, Data


class ZINC20(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, file_name="zinc20_graphs_100k"):
        self.file_name = file_name

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f"{self.file_name}.pth"

    @property
    def processed_file_names(self):
        return f"{self.file_name}.pth"

    def process(self):
        print("processing", self.raw_paths[0])
        graphs_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            graphs_list = [graph for graph in graphs_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graphs_list = [self.pre_transform(graph) for graph in graphs_list]

        torch.save(self.collate(graphs_list), self.processed_paths[0])

    def get(self, idx):
        data = super().get(idx)
        n_models = data.dockscores.size(0)
        n_atoms = data.pos_free.size(0)
        data.pos_docked = data.pos_docked.reshape(n_models, n_atoms, 3)
        return data

    @staticmethod
    def collate(graphs_list):
        keys = ['int_zinc_id', 'smiles', 'edge_index', 'z', 'dockscores', 'pos_free', 'pos_docked']
        keys_scalars = ['int_zinc_id', 'smiles']
        keys_index = ['edge_index']
        keys_regular_tensors = ['z', 'dockscores', 'pos_free']
        keys_irregular_tensors = ['pos_docked']

        # initialize data structure
        slices = {key: [0] for key in keys}
        data = Data()
        for key in keys:
            data[key] = []

        # fill data
        for graph in graphs_list:
            for key in keys:
                last_pos = slices[key][-1]
                if key in keys_irregular_tensors:
                    n_models, n_atoms, n_dim = graph[key].size()
                    data[key].append(graph[key].reshape(n_models * n_atoms, n_dim))
                    slices[key].append(last_pos + n_models * n_atoms)
                else:
                    data[key].append(graph[key])
                    if key in keys_scalars:
                        slices[key].append(last_pos + 1)
                    elif key in keys_index:
                        slices[key].append(last_pos + graph[key].size(1))
                    else:  # regular tensor
                        slices[key].append(last_pos + graph[key].size(0))

        # convert to suitable data type
        for key in keys:
            slices[key] = torch.tensor(slices[key], dtype=torch.long)
            if key in (keys_regular_tensors + keys_irregular_tensors):
                data[key] = torch.cat(data[key], dim=0)
            elif key in keys_index:
                data[key] = torch.cat(data[key], dim=1)
            else:  # keep 'scalars' as is (in lists)
                pass

        return data, slices
