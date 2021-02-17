from model_with_uncertainty.molecule_models import MolMCDropGNN
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

from torch.utils.data import Dataset
from LambdaZero.inputs import mol_to_graph
from rdkit import Chem

import os.path as osp
import torch_geometric.transforms as T
import LambdaZero.utils
import LambdaZero.inputs

from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolEnvGraph_v1
from LambdaZero.environments.reward import PredDockReward_v2
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyReward,DummyReward
from LambdaZero.contrib.model_with_uncertainty import MolMCDropGNN
from LambdaZero.contrib.oracle import DockingOracle
from LambdaZero.contrib.inputs import load_data_v1
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

import ray
ray.init()

def load_data_v1(target, dataset_split_path, dataset, dataset_config):
    dataset = dataset(**dataset_config)
    train_idxs, val_idxs, test_idxs = np.load(dataset_split_path, allow_pickle=True)
    data_list = []
    y_list = []
    for graph in dataset:
        y_list.append(getattr(graph,target))
        delattr(graph,target)
        delattr(graph, "pos")
        delattr(graph, "smiles")
        graph.x = torch.cat([graph.x, torch.zeros([graph.x.shape[0],2])],dim=1)
        graph.jbond_atmidx=torch.zeros([0, 2])
        graph.jbond_preds=torch.zeros([0, 0])
        graph.stem_atmidx=torch.zeros([0])
        graph.stem_preds=torch.zeros([0, 0])
        data_list.append(graph)

    train_x = [{"mol_graph":data_list[int(i)]} for i in train_idxs]
    train_y = [y_list[i] for i in train_idxs]
    val_x = [{"mol_graph":data_list[int(i)]} for i in val_idxs]
    val_y = [y_list[i] for i in val_idxs]
    return train_x, train_y, val_x, val_y

train_x, train_y, val_x, val_y = load_data_v1(
        'dockscore', 
        osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"), 
        LambdaZero.inputs.BrutalDock, 
        {
            "root": osp.join(datasets_dir, "brutal_dock/seh"),
            "props": ["dockscore", "smiles"],
            "transform": T.Compose([LambdaZero.utils.Complete(),
            LambdaZero.utils.Normalize("dockscore", -8.6, 1.1)]),
            "file_names": ["Zinc20_docked_neg_randperm_3k"],
        }
        )

model = MolMCDropGNN()
model.fit(train_x, train_y)
