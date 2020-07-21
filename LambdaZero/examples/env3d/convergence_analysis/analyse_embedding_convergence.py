"""
This script computes the convergence behavior of the minimum energy of a molecule with the number
of embedding conformations. The data is written to file to be visualized elsewhere.
"""
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, SanitizeMol, MolToSmiles
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag, Chem
from LambdaZero.environments.molMDP import MolMDP


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
        mol_with_hydrogen = Mol(mol)
        Chem.AddHs(mol_with_hydrogen)
        SanitizeMol(mol_with_hydrogen)
        AllChem.EmbedMultipleConfs(
            mol_with_hydrogen, numConfs=num_conf, randomSeed=random_seed
        )

        list_tuples = AllChem.MMFFOptimizeMoleculeConfs(
            mol_with_hydrogen, mmffVariant="MMFF94", maxIters=max_iters
        )
        list_energies = np.array([t[1] for t in list_tuples if t[0] == 0])

        number_of_successes = len(list_energies)
        if number_of_successes == 0:
            min_energy = np.NaN
        else:
            min_energy = np.min(list_energies)

        row = {
            "smiles": smiles,
            "num_conf": num_conf,
            "min_energy": min_energy,
            "successful_optimizations": number_of_successes,
        }
        list_rows.append(row)

    return pd.DataFrame(data=list_rows)


datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

results_dir = Path(summaries_dir).joinpath("env3d")

# computation parameters
number_of_blocks = 10
random_seed = 12312
list_num_conf = [1, 10, 25, 50, 100, 150, 200]
number_of_molecules = 100

results_path = Path(results_dir).joinpath(f'num_conf_convergence/conv_{number_of_blocks}_blocks.pkl')
results_path.parent.mkdir(exist_ok=True)

if __name__ == "__main__":

    np.random.seed(random_seed)
    molMDP = MolMDP(blocks_file=blocks_file)

    list_df = []
    for i in range(1, number_of_molecules+1):
        print(f"Molecule {i} of {number_of_molecules}")

        molMDP.reset()
        molMDP.random_walk(number_of_blocks)

        mol, mol_bond = mol_from_frag(
            jun_bonds=molMDP.molecule.jbonds, frags=molMDP.molecule.blocks, optimize=False
        )

        df = generate_minimum_energy_vs_number_of_configuration_profile(mol, list_num_conf, random_seed=random_seed)
        list_df.append(df)

    df = pd.concat(list_df).reset_index(drop=True)
    df.to_pickle(results_path)
