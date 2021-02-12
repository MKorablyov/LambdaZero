from torch.utils.data import Dataset
import time
import os.path as osp
import numpy as np
from LambdaZero.inputs import mol_to_graph
from rdkit import Chem
import torch

def load_data_v1(target, dataset_split_path, dataset, dataset_config):
    # make dataset

    # dockscore=[1],
    # edge_attr=[380, 4],
    # edge_index=[2, 380],
    # property=[1],
    # smiles="Cc1cccc(NC(=O)c2cccc3ccccc32)c1",
    # x=[20, 16])

    # edge_attr=[12, 4],
    # edge_index=[2, 12],
    # jbond_atmidx=[6, 2],
    # jbond_preds=[6, 0],
    # stem_atmidx=[20],
    # stem_preds=[20, 0],
    # x=[6, 16])

    dataset = dataset(**dataset_config)
    train_idxs, val_idxs, test_idxs = np.load(dataset_split_path, allow_pickle=True)

    # add dimension to the graph # todo: maybe do in inputs
    data_list = []
    for graph in dataset:
        # append stem and jbond features
        graph.x = torch.cat([graph.x, torch.zeros([graph.x.shape[0],2])],dim=1)
        data_list.append(graph)
        # delete coordinates
        delattr(graph, "pos")
        print("load data graph", graph, graph.property)

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

