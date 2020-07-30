import sys
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles
from tqdm import tqdm

from LambdaZero.examples.env3d.rdkit_utilities import get_lowest_energy_and_mol_with_conformer


def generate_minimum_energy_vs_number_of_configuration_profile(
    mol: Mol, list_num_conf: List[int], max_iters: int = 500, random_seed: int = 0
):
    """
    Create the minimum energy obtained from a set of num_conf embeddings.
    Args:
        mol (Mol): a Mol object
        list_num_conf (List[int]): the list of num_conf to try
        max_iters (int): maximum number of iterations for the optimizer to converge
        random_seed (int):  random seed for the embedding

    Returns:
        df (pandas.DataFrame): dataframe with the results

    """

    smiles = MolToSmiles(mol)

    list_rows = []
    for num_conf in tqdm(list_num_conf, desc="EMBED_CONFIG", file=sys.stdout):
        try:
            min_energy, mol_with_hydrogen, number_of_successes = get_lowest_energy_and_mol_with_conformer(
                mol, num_conf, max_iters, random_seed
            )
        except ValueError as e:
            min_energy = np.NaN
            number_of_successes = 0
            print(f"\n{e}\n")

        row = {
            "smiles": smiles,
            "num_conf": num_conf,
            "min_energy": min_energy,
            "successful_optimizations": number_of_successes,
        }
        list_rows.append(row)

    return pd.DataFrame(data=list_rows)