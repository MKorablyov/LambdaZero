"""
This script plots the convergence data of min_energy vs. num_conf. It assumes that the data has been generated
with another script and is written to file as a pandas pickle.
"""
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.5)

import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

import pandas as pd

from LambdaZero.utils import get_external_dirs

_, _, summaries_dir = get_external_dirs()
# results_path = Path(summaries_dir).joinpath(f'num_conf_convergence/conv_{number_of_blocks}_blocks_test.pkl')

# computation parameters
number_of_blocks = 5
results_dir = Path(summaries_dir).joinpath("env3d")
results_path = Path(results_dir).joinpath(
    f"num_conf_convergence/conv_{number_of_blocks}_blocks.pkl"
)

if __name__ == "__main__":

    df = pd.read_pickle(results_path)

    _groups = df.groupby(by="smiles")

    df["relative energy"] = df["min_energy"] - _groups["min_energy"].transform("mean")
    df["energy change"] = _groups["min_energy"].transform(
        lambda x: np.diff(x, prepend=np.NaN)
    )
    df["relative energy change"] = _groups["min_energy"].transform(
        lambda x: np.diff(x, prepend=np.NaN) / x
    )
    groups = df.groupby(by="smiles")
    number_of_molecules = len(groups)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(
        f"Energy distribution for a random sample of {number_of_molecules} molecules with {number_of_blocks} blocks"
    )

    list_minimum_energies = groups['min_energy'].apply(np.nanmin).values
    ax = fig.add_subplot(111)
    ax.hist(list_minimum_energies)
    ax.set_xlabel('energy (kcal/mol)')
    ax.set_ylabel('counts')
    fig.savefig(results_dir.joinpath(f"energy_histogram_{number_of_blocks}_blocks.png"))

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(
        f"Energy behavior vs. number of embeddings \nfor a random sample of {number_of_molecules} molecules with {number_of_blocks} blocks"
    )

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    for smiles in groups.groups.keys():
        group_df = groups.get_group(smiles)
        list_num_conf = group_df["num_conf"].values
        list_min_energies = group_df["min_energy"].values

        min_energy = np.nanmin(list_min_energies)
        print(f"min energy: {min_energy}")

        list_relative_energy_change = group_df["relative energy change"].values
        list_energy_change = group_df["energy change"].values
        list_relative_energies = group_df["relative energy"].values

        ax1.plot(list_num_conf, list_min_energies, "-", label=smiles)
        ax1.set_ylabel("energy (kcal/mol)")

        ax2.plot(list_num_conf, list_relative_energies, "-", label=smiles)
        ax2.set_ylabel("energy - <energy> (kcal/mol)")

        ax3.plot(list_num_conf, list_energy_change, "-", label=smiles)
        ax3.set_ylabel("change in energy (kcal/mol)")

        ax4.plot(list_num_conf, list_relative_energy_change, "-", label=smiles)
        ax4.set_ylabel("relative change in energy (%)")

    ax3.set_xlabel("number of configuration")
    ax4.set_xlabel("number of configuration")

    fig.savefig(results_dir.joinpath(f"statistics_{number_of_blocks}_blocks.png"))
    plt.show()
