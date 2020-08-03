"""
The goal of this script is to analyse the toy dataset, plotting the most frequent blocks to
sanity check we get reasonable molecules.
"""
import json
import os
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles

from LambdaZero.examples.env3d.utilities import get_angles_in_degrees
from LambdaZero.utils import get_external_dirs


sns.set(font_scale=1.5)

datasets_dir, _, summaries_dir = get_external_dirs()


block_file_path_path = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

number_of_parent_blocks = 5


dataset_base_path = Path(summaries_dir).joinpath("env3d/dataset/from_cluster/RUN4/")
dataset_path = dataset_base_path.joinpath("combined_dataset.feather")
results_dir = dataset_base_path.joinpath("analysis/")
results_dir.mkdir(exist_ok=True)

if __name__ == "__main__":

    with open(block_file_path_path, 'r') as f:
        blocks_dict = json.load(f)

    vocabulary = blocks_dict['block_smi']
    vocabulary_size = len(vocabulary)

    df = pd.read_feather(dataset_path)

    number_of_blocks_present = np.unique(df['attachment_block_index']).shape[0]
    subtitle = f"Vocabulary size: {vocabulary_size}, " \
               f"number of blocks with non-zero count: {number_of_blocks_present}, " \
               f"Number of molecules: {len(df)}"

    #  Plot the distribution of blocks and angles
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Basic analysis of dataset. \n{subtitle}")

    ax1 = fig.add_subplot(121)
    ax1.set_title("child block index")

    colors = []
    for block_index in range(105):
        number_of_atoms = MolFromSmiles(vocabulary[f"{block_index}"]).GetNumAtoms()
        if number_of_atoms == 1:
            colors.append('red')
        else:
            colors.append('blue')

    value_count_series = pd.Series(-1, index=range(105))
    actual_values = df['attachment_block_index'].value_counts().sort_index()
    value_count_series[actual_values.index] = actual_values

    value_count_series.plot.bar(width=0.9, ax=ax1, color=colors)

    # place a text box in upper left in axes coords
    textstr = "Red means single atom block"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.4, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax1.set_xlabel('block index')
    ax1.set_ylabel('count')
    ax1.set_xlim(0, 105)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(122)
    ax2.set_title("Attachment angle")

    angles_in_degree = get_angles_in_degrees(df['attachment_angle'].values)
    ax2.hist(angles_in_degree, bins=60)
    ax2.set_xlabel('angle (degree)')
    ax2.set_xlim(-60, 360)
    ax2.set_ylabel('count')

    # place a text box in upper left in axes coords
    textstr = "Negative angle means single atom child block"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(0.05, 0.5, textstr, transform=ax2.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.semilogy()
    fig.savefig(results_dir.joinpath("small_dataset_block_and_angle_distribution.png"))

    #  Plot the sorted count distribution and cumulative sum; is there a natural cutoff?
    value_count_series = df['attachment_block_index'].value_counts()
    cumulative_counts_series = value_count_series.cumsum()

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Sorted counts and cumulative sum over blocks\n{subtitle}")
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    value_count_series.plot.bar(width=0.9, ax=ax1)
    cumulative_counts_series.reset_index(drop=True).plot.line(drawstyle='steps', color='orange', lw=4, ax=ax2)
    ax1.set_xticklabels([])
    ax1.set_xlabel('block index sorted by count')
    ax1.set_ylabel('block count')
    ax2.set_ylabel('count cumulative sum')
    ax1.grid(False)
    ax2.grid(False)
    ax1.semilogy()
    fig.savefig(results_dir.joinpath("small_dataset_cumulative_counts.png"))

    #  Plot the energy distributions
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Energy distributions\n{subtitle}")
    ax1 = fig.add_subplot(111)
    df[['total_energy', 'parent_energy', 'binding_energy']].plot.hist(bins=100, alpha=0.5, ax=ax1)
    ax1.set_xlabel('Energy (kcal/mol)')
    fig.savefig(results_dir.joinpath("small_dataset_energies_distribution.png"))

    # Plot the most frequent blocks
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
