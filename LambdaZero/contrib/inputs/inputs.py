import time
import os.path as osp
import numpy as np
from LambdaZero.inputs import mol_to_graph
from rdkit import Chem

def load_data_v1(target, dataset_split_path, dataset, dataset_config):
    # make dataset
    dataset = dataset(**dataset_config)
    train_idxs, val_idxs, test_idxs = np.load(dataset_split_path, allow_pickle=True)
    # take corresponding indices from data loaders
    train_x = [{"mol_graph":dataset[int(i)]} for i in train_idxs]
    train_y = [getattr(dataset[int(i)],target) for i in train_idxs]
    val_x = [{"mol_graph":dataset[int(i)]} for i in val_idxs]
    val_y = [getattr(dataset[int(i)],target) for i in val_idxs]
    return train_x, train_y, val_x, val_y


class Mol2GraphProc:
    def __init__(self, props, transform):
        #self.props = props
        self.transform = transform

    def __call__(self, molecules):
        pr_molecules = []
        for molecule in molecules:
            smi = Chem.MolToSmiles(molecule.mol)
            #props = {prop:getattr(molecule, prop) for prop in self.props}
            graph = mol_to_graph(smi, {"dockscore":1.0}) # fixme!!!!!!!!
            graph = self.transform(graph)
            pr_molecules.append({"molecule": molecule, "graph": graph})
        return pr_molecules

# def _brutal_dock_proc(smi, props, pre_filter, pre_transform):
#     try:
#         graph = mol_to_graph(smi, props)
#     except Exception as e:
#         return None
#     if pre_filter is not None and not pre_filter(graph):
#         return None
#     if pre_transform is not None:
#         graph = pre_transform(graph)
#     return graph