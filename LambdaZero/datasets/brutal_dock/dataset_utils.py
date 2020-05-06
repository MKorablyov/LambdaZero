from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch_geometric.data import DataLoader

from LambdaZero.chem import mol_to_graph


def get_smiles_and_scores_from_feather(feather_data_path: Path) -> Tuple[List[str], List[float]]:
    df = pd.read_feather(feather_data_path)
    list_smiles = list(df['smiles'].values)
    list_scores = list(df['gridscore'].to_numpy())
    return list_smiles, list_scores


def get_molecule_graphs_from_smiles_and_scores(list_smiles: List[str], list_scores: List[float]):
    list_graphs = [mol_to_graph(smiles, dockscore=score) for (smiles, score) in zip(list_smiles, list_scores)]
    return list_graphs


def get_scores_statistics(training_dataloader: DataLoader):

    number_of_graphs = len(training_dataloader.dataset)

    with torch.no_grad():
        score_sum = torch.zeros(1)
        score_square_sum = torch.zeros(1)

        for batch in training_dataloader:
            score_sum += torch.sum(batch.dockscore)
            score_square_sum += torch.sum(batch.dockscore ** 2)

        mean_tensor = score_sum/number_of_graphs
        std_tensor = torch.sqrt(score_square_sum/number_of_graphs - mean_tensor**2)

        mean = mean_tensor.item()
        std = std_tensor.item()

        return mean, std

