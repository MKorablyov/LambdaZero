"""
This script computes the convergence behavior of the minimum energy of a molecule with the number
of embedding conformations. The data is written to file to be visualized elsewhere.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import LambdaZero.utils
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.convergence_analysis.utils import \
    generate_minimum_energy_vs_number_of_configuration_profile

datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

results_dir = Path(summaries_dir).joinpath("env3d")

# computation parameters
number_of_blocks = 5
random_seed = 12312
list_num_conf = [1, 10, 25, 50, 100, 150, 200]
number_of_molecules = 100

results_path = Path(results_dir).joinpath(
    f"num_conf_convergence/conv_{number_of_blocks}_blocks.pkl"
)
results_path.parent.mkdir(exist_ok=True)

if __name__ == "__main__":

    np.random.seed(random_seed)
    molMDP = MolMDP(blocks_file=blocks_file)

    list_df = []
    for i in range(1, number_of_molecules + 1):
        print(f"Molecule {i} of {number_of_molecules}")

        molMDP.reset()
        molMDP.random_walk(number_of_blocks)
        mol = molMDP.molecule.mol

        df = generate_minimum_energy_vs_number_of_configuration_profile(
            mol, list_num_conf, random_seed=random_seed
        )
        list_df.append(df)

    df = pd.concat(list_df).reset_index(drop=True)
    df.to_pickle(results_path)
