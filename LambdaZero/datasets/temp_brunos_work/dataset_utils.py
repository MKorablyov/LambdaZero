import sys

import pandas as pd
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from LambdaZero.chem import mol_to_graph


def get_molecule_graphs_from_raw_data_dataframe(raw_data_df: pd.DataFrame):
    list_graphs = []
    for _, row in tqdm(raw_data_df.iterrows(), desc="MOL_TO_GRAPH", file=sys.stdout):
        smiles = row["smiles"]
        graph = mol_to_graph(smiles,
                             gridscore=row.get("gridscore", None),
                             dockscore=row.get("dockscore", None),
                             klabel=row.get("klabel", None))
        graph.smiles = smiles
        list_graphs.append(graph)
    return list_graphs


def get_scores_statistics(training_dataloader: DataLoader):

    number_of_graphs = len(training_dataloader.dataset)

    with torch.no_grad():
        score_sum = torch.zeros(1)
        score_square_sum = torch.zeros(1)

        for batch in training_dataloader:
            score_sum += torch.sum(batch.gridscore)
            score_square_sum += torch.sum(batch.gridscore ** 2)

        mean_tensor = score_sum/number_of_graphs
        std_tensor = torch.sqrt(score_square_sum/number_of_graphs - mean_tensor**2)

        mean = mean_tensor.item()
        std = std_tensor.item()

        return mean, std

