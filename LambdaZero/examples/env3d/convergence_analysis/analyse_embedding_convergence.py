"""
This script computes the convergence behavior of the minimum energy of a molecule with the number
of embedding conformations. The data is written to file to be visualized elsewhere.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, MolToSmiles
from rdkit.Chem.rdmolops import SanitizeMol
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.environments.molMDP import MolMDP


def get_cumulative_converged_min(list_tuples_energy_converged):
    converged_min = np.inf
    list_cumulative_converged_min = []

    for (c, e) in list_tuples_energy_converged:
        if c == 0:
            if e < converged_min:
                converged_min = e
        list_cumulative_converged_min.append(converged_min)
    return list_cumulative_converged_min


datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

results_dir = Path(summaries_dir).joinpath("env3d")

# computation parameters
number_of_blocks = 5

random_seed = 123
small_num_conf = 5
large_num_conf = 100

number_of_molecules = 100
max_iters = 200

results_path = Path(results_dir).joinpath(
    f"num_conf_convergence/convergence_{number_of_blocks}_blocks.pkl"
)
results_path.parent.mkdir(exist_ok=True)

if __name__ == "__main__":

    np.random.seed(random_seed)
    reference_molMDP = MolMDP(blocks_file=blocks_file)

    # find a few molecules that embed quickly
    print("Finding reference molecules that embed quickly")
    list_reference_mols = []
    for _ in tqdm(range(number_of_molecules)):
        number_of_successes = 0
        counter = 0
        while number_of_successes == 0:
            counter += 1
            print(f"  - Attempt {counter}")
            reference_molMDP.reset()
            reference_molMDP.random_walk(number_of_blocks)
            augmented_mol = Chem.AddHs(reference_molMDP.molecule.mol)
            SanitizeMol(augmented_mol)
            AllChem.EmbedMultipleConfs(
                augmented_mol, numConfs=small_num_conf, randomSeed=random_seed
            )
            list_tuples_energy_converged = AllChem.MMFFOptimizeMoleculeConfs(
                augmented_mol, mmffVariant="MMFF94", maxIters=max_iters
            )
            number_of_successes = np.sum(
                [t[0] == 0 for t in list_tuples_energy_converged]
            )
        list_reference_mols.append(Mol(reference_molMDP.molecule.mol))

    print("Generating embedding energy data")
    list_df = []
    list_num_conf = list(range(large_num_conf))
    for reference_mol in tqdm(list_reference_mols):
        augmented_mol = Chem.AddHs(reference_mol)
        SanitizeMol(augmented_mol)
        AllChem.EmbedMultipleConfs(
            augmented_mol, numConfs=large_num_conf, randomSeed=random_seed
        )
        list_tuples_energy_converged = AllChem.MMFFOptimizeMoleculeConfs(
            augmented_mol, mmffVariant="MMFF94", maxIters=max_iters
        )
        list_cumulative_converged_min = get_cumulative_converged_min(
            list_tuples_energy_converged
        )

        sub_df = pd.DataFrame(
            data={
                "min_energy": list_cumulative_converged_min,
                "num_conf": list_num_conf,
            }
        )
        sub_df["smiles"] = MolToSmiles(reference_mol)
        list_df.append(sub_df)

        # Write at every iteration because it is SLOW to generate this data
        convergence_df = pd.concat(list_df).reset_index(drop=True)
        convergence_df.to_pickle(results_path)
