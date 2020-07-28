"""
The goal of this script is to analyse the toy dataset, plotting the most frequent blocks to
sanity check we get reasonable molecules.
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from LambdaZero.examples.env3d.utilities import get_angles_in_degrees
from LambdaZero.utils import get_external_dirs

datasets_dir, _, summaries_dir = get_external_dirs()
results_dir = Path(summaries_dir).joinpath("env3d/analysis/")
results_dir.mkdir(exist_ok=True)

block_file_path_path = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

number_of_parent_blocks = 5

dataset_path = Path(summaries_dir).joinpath("env3d/dataset/").joinpath(
    f"env3d_dataset_{number_of_parent_blocks}_parent_blocks.feather"
)

if __name__ == "__main__":

    df = pd.read_feather(dataset_path)

    # Plot the most frequent blocks

    with open(block_file_path_path, 'r') as f:
        blocks_dict = json.load(f)

    block_indices, counts = np.unique(df['attachment_block_index'].values, return_counts=True)
    sorting_indices = np.argsort(counts)[::-1]
    block_indices = block_indices[sorting_indices]
    counts = counts[sorting_indices]

    list_common_block_smiles = [blocks_dict['block_smi'][f'{block_indices[i]}'] for i in range(10)]
    list_common_blocks = [Chem.MolFromSmiles(smiles) for smiles in list_common_block_smiles]

    img = Draw.MolsToGridImage(list_common_blocks, molsPerRow=5, subImgSize=(200, 200),
                               legends=[f'block {index}, count = {count}' for index, count in zip(block_indices, counts)])

    img.save(results_dir.joinpath("common_blocks.png"))

    # Plot sample of child molecules
    number_of_sample_molecules = 25
    list_smiles = df['smi'].values
    list_mols = [Chem.MolFromSmiles(smiles) for smiles in list_smiles[:number_of_sample_molecules]]
    list_child_blocks = df['attachment_block_index'].values[:number_of_sample_molecules]

    img = Draw.MolsToGridImage(list_mols, molsPerRow=5, subImgSize=(200, 200),
                               legends=[f'child block {index}' for index in list_child_blocks])
    img.save(results_dir.joinpath("sample_of_molecules.png"))

    #  Plot the distribution of blocks and angles
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Basic analysis of dataset. Number of molecules: {len(df)}")

    ax1 = fig.add_subplot(211)
    ax1.set_title("child block index")
    df['attachment_block_index'].hist(bins=np.arange(106), ax=ax1)
    ax1.set_xlabel('block index')
    ax1.set_ylabel('count')

    ax2 = fig.add_subplot(212)
    ax2.set_title("Attachment angle")

    angles_in_degree = get_angles_in_degrees(df['attachment_angle'].values)
    ax2.hist(angles_in_degree, bins=60)
    ax2.set_xlabel('angle (degree)')
    ax2.set_xlim(0, 360)
    ax2.set_ylabel('count')
    fig.savefig(results_dir.joinpath("small_dataset_block_and_angle_distribution.png"))
