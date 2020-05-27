import sys

from chemprop.features import MolGraph
import pandas as pd
from tqdm import tqdm


def get_chemprop_graphs_from_raw_data_dataframe(raw_data_df: pd.DataFrame):
    list_data = []

    for _, row in tqdm(raw_data_df.iterrows(), total=len(raw_data_df), desc="CHEMPROP_GRAPH", file=sys.stdout):
        smiles = row["smiles"]

        mol_graph = MolGraph(smiles)

        data = dict(mol_graph=mol_graph,
                    gridscore=row.get("gridscore", None),
                    klabel=row.get("klabel", None))

        list_data.append(data)
    return list_data