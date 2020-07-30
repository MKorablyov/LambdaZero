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
from LambdaZero.examples.env3d.dataset.data_generation import get_blocks_embedding_energies, \
    compute_parent_and_all_children_energies

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

results_dir = Path(summaries_dir).joinpath("env3d")

# computation parameters
number_of_blocks = 5

random_seed = 12312
list_num_conf = [10, 20, 40]
number_of_molecules = 100

results_path = Path(results_dir).joinpath(
    f"num_conf_convergence/convergence_{number_of_blocks}_blocks.pkl"
)
results_path.parent.mkdir(exist_ok=True)


max_iters = 200

if __name__ == "__main__":

    np.random.seed(random_seed)
    child_block_energies_dict = get_blocks_embedding_energies(blocks_file)

    reference_molMDP = MolMDP(blocks_file=blocks_file)

    list_df = []
    for _ in range(number_of_molecules):

        reference_molMDP.reset()
        reference_molMDP.random_walk(number_of_blocks)

        number_of_stems = len(reference_molMDP.molecule.stems)
        if number_of_stems < 1:
            continue

        attachment_stem_idx = np.random.choice(number_of_stems)

        for num_conf in list_num_conf:
            try:
                df, _ = compute_parent_and_all_children_energies(
                    reference_molMDP,
                    attachment_stem_idx,
                    child_block_energies_dict,
                    num_conf,
                    max_iters,
                    random_seed)

                df['num_conf'] = num_conf
                list_df.append(df)
            except ValueError as e:
                print("parent failed. Continue")
                continue

    convergence_df = pd.concat(list_df).reset_index(drop=True)
    convergence_df.to_pickle(results_path)
