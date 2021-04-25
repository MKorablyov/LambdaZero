import os, time, os.path as osp
from itertools import repeat, product


import torch
from torch.utils.data import Dataset
import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

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


class ListGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graps = graphs

    def __getitem__(self, idx):
        return self.graps[idx]

    def __len__(self):
        return len(self.graps)


