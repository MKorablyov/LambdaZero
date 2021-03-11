from torch.utils.data import Dataset
import time
import os, os.path as osp
import numpy as np
from LambdaZero.inputs import mol_to_graph

from rdkit import Chem
import torch
import ray
import pandas as pd
from itertools import repeat, product

from LambdaZero.environments import BlockMoleculeData, GraphMolObs


def collate(data_list):
    r"""Collates a python list of data objects to the internal storage
    format of :class:`torch_geometric.data.InMemoryDataset`."""
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(
                item.__cat_dim__(key, item[key]))
        else:
            s = slices[key][-1] + 1
        slices[key].append(s)

    if hasattr(data_list[0], '__num_nodes__'):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        item = data_list[0][key]
        if torch.is_tensor(item) and len(data_list) > 1:
            data[key] = torch.cat(data[key],
                                  dim=data.__cat_dim__(key, item))
        elif torch.is_tensor(item):  # Don't duplicate attributes...
            data[key] = data[key][0]
        elif isinstance(item, int) or isinstance(item, float):
            data[key] = torch.tensor(data[key])

        slices[key] = torch.tensor(slices[key], dtype=torch.long)
    return data, slices

def separate(data_, slices_):
    num_graphs = len([x for x in slices_.values()][0])-1
    data_list = []
    for idx in range(num_graphs):
        data = data_.__class__()
        if hasattr(data_, '__num_nodes__'):
            data.num_nodes = data_.__num_nodes__[idx]
        for key in data_.keys:
            item, slices = data_[key], slices_[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[data_.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        data_list.append(data)
    return data_list

@ray.remote
def obs_from_smi(smi):
    molecule = BlockMoleculeData()
    molecule._mol = Chem.MolFromSmiles(smi)
    graph, _ = GraphMolObs()(molecule)
    return graph

def temp_load_data_v1(mean, std, dataset_split_path, raw_path, proc_path, file_names):

    if not all([osp.exists(osp.join(proc_path, file_name + ".pt")) for file_name in file_names]):
        print("processing graphs from smiles")
        if not osp.exists(proc_path): os.makedirs(proc_path)

        for file_name in file_names:
            docked_index = pd.read_feather(osp.join(raw_path, file_name + ".feather"))
            y = list(((mean - docked_index["dockscore"].to_numpy(dtype=np.float32)) / std))

            smis = docked_index["smiles"].tolist()
            graphs = ray.get([obs_from_smi.remote(smi) for smi in smis])
            # save graphs
            data, slices = collate(graphs)
            torch.save((data, slices, y), osp.join(proc_path, file_name + ".pt"))

    graph_list, y_list = [], []
    for file_name in file_names:
        data,slices, y = torch.load(osp.join(proc_path, file_name + ".pt"))
        graphs = separate(data, slices)
        graph_list.extend(graphs)
        y_list.extend(y)
    # split into train test sets
    train_idxs, val_idxs, test_idxs = np.load(dataset_split_path, allow_pickle=True)
    train_x = [{"mol_graph":graph_list[i]} for i in train_idxs]
    train_y = [y_list[i] for i in train_idxs]
    val_x = [{"mol_graph":graph_list[i]} for i in val_idxs]
    val_y = [y_list[i] for i in val_idxs]
    return train_x, train_y, val_x, val_y



class ListGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graps = graphs
        # todo use torch geometric to aggregate graphs together
        # use torch_geometric slices on each batch

    def __getitem__(self, idx):
        return self.graps[idx]

    def __len__(self):
        return len(self.graps)

