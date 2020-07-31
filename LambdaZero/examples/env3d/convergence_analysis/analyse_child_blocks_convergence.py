"""
This script analyses the component blocks and computes the convergence behavior of the minimum energy
of embedding these blocks on their own. The output is a combination of print to console and plots.

The main conclusions are:

1.
 - Some 1 atom blocks have zero energy, embedded with or without hydrogen (S, O, N, C [not zero but very close])
 - Some 1 atom blocks cannot be embedded with hydrogen (Br, Cl, F), and have zero energy when embedded with hydrogen.
 - Iodine (I) cannot be embedded with or without hydrogen.
    ---> it makes sense to just put the energy of one-atom block to zero.

2. The embedding energy of blocks converges very fast with num_conf. By num_conf = 25, the error is negligible for all
   blocks.

3. Some distinct blocks have the same smiles. The block_r datastructure indicates HOW the block should be connected.
   It makes sense to consider these as distinct "blocks" from the point of view of classification, but they should have
   the same energy.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem.rdmolfiles import MolFromSmiles

import LambdaZero.utils
from LambdaZero.examples.env3d.convergence_analysis.utils import (
    generate_minimum_energy_vs_number_of_configuration_profile,
)
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_lowest_energy_and_mol_with_conformer,
)

datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

results_dir = Path(summaries_dir).joinpath("env3d")

# computation parameters
random_seed = 12312
list_num_conf = [1, 10, 25, 50, 100]

max_iters = 200

if __name__ == "__main__":
    np.random.seed(random_seed)

    blocks_df = pd.read_json(blocks_file)
    blocks_df["number_of_atoms"] = blocks_df['block_smi'].apply(lambda smiles: MolFromSmiles(smiles).GetNumAtoms())
    number_of_distinct_smiles = len(np.unique(blocks_df["block_smi"]))

    # Show that the block_r lists are NOT permutations of each other for multi-occurring smiles
    groups = blocks_df.groupby(by="block_smi")

    for smiles, group_df in groups:
        if len(group_df) == 1:
            continue
        representative_block_r_set = set(group_df["block_r"].iloc[0])
        for block_r in group_df["block_r"]:
            if representative_block_r_set != set(block_r):
                print("Blocks are not the same")
                print(group_df)
                continue

    # Check energy convergence of embedded blocks
    list_energy_convergence_df = []

    energy_rows = []
    for smiles, group_df in groups:
        block_mol = MolFromSmiles(smiles)
        df = generate_minimum_energy_vs_number_of_configuration_profile(
            block_mol, list_num_conf, max_iters=max_iters, random_seed=random_seed
        )
        list_energy_convergence_df.append(df)

        try:
            energy_without_h, _, _ = get_lowest_energy_and_mol_with_conformer(
                block_mol,
                list_num_conf[-1],
                max_iters=max_iters,
                random_seed=random_seed,
                augment_mol_with_hydrogen=False,
            )
        except Exception as e:
            energy_without_h = np.NaN

        energy_with_h = df[df["num_conf"] == list_num_conf[-1]]["min_energy"].values[0]
        energy_rows.append(
            {
                "smiles": smiles,
                "number of atoms": block_mol.GetNumAtoms(),
                "energy with h": energy_with_h,
                "energy without h": energy_without_h,
            }
        )

    energy_hydrogen_df = pd.DataFrame(energy_rows)
    energy_convergence_df = pd.concat(list_energy_convergence_df).reset_index(drop=True)

    print("Single atom block energies:")
    print(energy_hydrogen_df[energy_hydrogen_df['number of atoms'] == 1])

    print("Smiles with NaN energies:")
    print(energy_hydrogen_df[energy_hydrogen_df['energy with h'].isna()])

    # plot convergence with number of conformers. By num_conf = 25, the error is smaller than 10^{-6}!
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    fig.suptitle(f"Convergence of block energies with number of embedding conf. max iters = {max_iters}")
    ax.set_xlabel("num conf")
    ax.set_ylabel("Energy difference with most converged (kcal/mol)")

    for smiles, group_df in energy_convergence_df.groupby(by='smiles'):
        x = group_df['num_conf'].values
        y = group_df['min_energy'].values
        sorting_indices = np.argsort(x)
        x = x[sorting_indices]
        y = y[sorting_indices]

        error = np.abs(y-y[-1])

        ax.semilogy(x, error, '-')

    plt.show()
