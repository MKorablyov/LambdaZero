from torch.utils.data import Dataset
import time
import os.path as osp
import numpy as np
from LambdaZero.inputs import mol_to_graph
from rdkit import Chem
import torch

def temp_load_data_v1(mean, std, dataset_split_path, dataset, dataset_config):
    " "
    # make dataset
    dataset = dataset(**dataset_config)

    train_idxs, val_idxs, test_idxs = np.load(dataset_split_path, allow_pickle=True)
    # add dimension to the graph # todo: maybe do in inputs
    data_list = []
    y_list = []
    for graph in dataset:
        y_list.append(getattr(graph,"dockscore"))
        delattr(graph,"dockscore")
        # append stem and jbond features
        delattr(graph, "pos")
        delattr(graph, "smiles")
        graph.x = torch.cat([graph.x, torch.zeros([graph.x.shape[0],2])],dim=1)
        graph.jbond_atmidx=torch.zeros([0, 2])
        graph.jbond_preds=torch.zeros([0, 0])
        graph.stem_atmidx=torch.zeros([0])
        graph.stem_preds=torch.zeros([0, 0])
        data_list.append(graph)

    y_list = [(mean - y) / std for y in y_list] # this flips the dockscore to norm version
    train_x = [{"mol_graph":data_list[int(i)]} for i in train_idxs]
    train_y = [y_list[i] for i in train_idxs]
    val_x = [{"mol_graph":data_list[int(i)]} for i in val_idxs]
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

