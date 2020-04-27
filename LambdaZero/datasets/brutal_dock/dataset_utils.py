from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from LambdaZero.chem import mol_to_graph
from LambdaZero.datasets.brutal_dock import BRUTAL_DOCK_DATA_DIR

d4_feather_data_path = BRUTAL_DOCK_DATA_DIR.joinpath("d4/raw/dock_blocks105_walk40_clust.feather")
load_model_path = BRUTAL_DOCK_DATA_DIR.joinpath("d4/dock_blocks105_walk40_12_clust_model002")


def get_smiles_and_scores_from_feather(feather_data_path: Path):
    df = pd.read_feather(feather_data_path)
    list_smiles = df['smiles'].values
    list_scores = df['gridscore'].to_numpy()
    return list_smiles, list_scores


def get_molecule_graph_dataset(list_smiles: List[str], list_scores: List[float]):
    list_graphs = [mol_to_graph(smiles, dockscore=score) for (smiles, score) in zip(list_smiles, list_scores)]
    return list_graphs


def get_scores_statistics(list_graphs):
    list_scores = [g.dockingscore for g in list_graphs]
    mean = np.mean(list_scores)
    std = np.std(list_scores)
    return mean, std


