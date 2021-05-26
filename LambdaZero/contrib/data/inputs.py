import time
import os, os.path as osp
import numpy as np

from rdkit import Chem
import torch
import ray
import pandas as pd
import LambdaZero.utils
from LambdaZero.environments import BlockMoleculeData, GraphMolObs
import LambdaZero.contrib.functional

from .data import collate, separate


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()




@ray.remote
def obs_from_smi(smi,):
    # fixme this does not allow me to change default molecule encoding/decoding parameters
    molecule = BlockMoleculeData()
    molecule._mol = Chem.MolFromSmiles(smi)
    graph, _ = GraphMolObs()(molecule)
    return graph

def temp_load_data(mean, std, act_y, dataset_split_path, raw_path, proc_path, file_names):
    if not all([osp.exists(osp.join(proc_path, file_name + ".pt")) for file_name in file_names]):
        print("processing graphs from smiles, this might take some time..")
        if not osp.exists(proc_path): os.makedirs(proc_path)
        for file_name in file_names:
            docked_index = pd.read_feather(osp.join(raw_path, file_name + ".feather"))

            if mean == None and std == None and act_y == None:
                y = docked_index["norm_dockscore"].to_list()
                print("using norm dockscores from the dataset")
            else:
                y = list(((mean - docked_index["dockscore"].to_numpy(dtype=np.float32)) / std))
                y = act_y(y) # apply soft negatives
                print("using original dockscores in the dataset and normalizing them")

            smis = docked_index["smiles"].tolist()
            graphs = ray.get([obs_from_smi.remote(smi) for smi in smis])
            # save graphs
            data, slices = collate(graphs)
            torch.save((data, slices, y), osp.join(proc_path, file_name + ".pt"))

    smis, graph_list, y_list = [], [], []
    for file_name in file_names:
        # load smiles from raw data
        docked_index = pd.read_feather(osp.join(raw_path, file_name + ".feather"))
        smis.extend(docked_index["smiles"].tolist())
        # load processed graphs
        data, slices, y = torch.load(osp.join(proc_path, file_name + ".pt"))
        graphs = separate(data, slices)
        graph_list.extend(graphs)
        y_list.extend(y)

    # split into train test sets
    train_idxs, val_idxs, _ = np.load(dataset_split_path, allow_pickle=True)
    train_x = [{"smiles":smis[i],"mol_graph":graph_list[i]} for i in train_idxs]
    train_y = [y_list[i] for i in train_idxs]
    val_x = [{"smiles":smis[i],"mol_graph":graph_list[i]} for i in val_idxs]
    val_y = [y_list[i] for i in val_idxs]
    return train_x, train_y, val_x, val_y

config_temp_load_data_v1 = {
    "mean": -8.6,
    "std": 1.1,
    "act_y": LambdaZero.contrib.functional.elu2,
    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
    "raw_path": osp.join(datasets_dir, "brutal_dock/seh/raw"),
    "proc_path": osp.join(datasets_dir, "brutal_dock/seh/processed_rlbo"),
    "file_names": ["Zinc20_docked_neg_randperm_3k"],
}

config_temp_load_data_v2 = {
    "mean": None, "std": None, "act_y": None,
    "raw_path": osp.join(datasets_dir, "brutal_dock/seh/raw"),
    "proc_path": osp.join(datasets_dir, "brutal_dock/seh/processed_rlbo"),
    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/random_molecule_proxy_20k.npy"),
    "file_names": ["random_molecule_proxy_20k"],
}