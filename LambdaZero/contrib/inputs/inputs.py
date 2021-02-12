from torch.utils.data import Dataset
import time
import os.path as osp
import numpy as np
from LambdaZero.inputs import mol_to_graph
from rdkit import Chem
import torch

def load_data_v1(target, dataset_split_path, dataset, dataset_config):
    # make dataset
    dataset = dataset(**dataset_config)
    train_idxs, val_idxs, test_idxs = np.load(dataset_split_path, allow_pickle=True)

    # add dimension to the graph # todo: maybe do in inputs
    data_list = []
    for graph in dataset:
        graph.x = torch.cat([graph.x, torch.zeros([graph.x.shape[0],2])],dim=1)
        data_list.append(graph)

    train_x = [{"mol_graph":data_list[int(i)]} for i in train_idxs]
    train_y = [getattr(data_list[int(i)],target) for i in train_idxs]
    val_x = [{"mol_graph":data_list[int(i)]} for i in val_idxs]
    val_y = [getattr(data_list[int(i)],target) for i in val_idxs]

    return train_x, train_y, val_x, val_y


class ListGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graps = graphs

    def __getitem__(self, idx):
        return self.graps[idx]

    def __len__(self):
        return len(self.graps)

