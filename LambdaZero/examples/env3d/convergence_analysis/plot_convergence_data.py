"""
This script plots the convergence data of min_energy vs. num_conf. It assumes that the data has been generated
with another script, convergence_analysis/analyse_embedding_convergence.py, and is written to file as a pandas pickle.
"""
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from LambdaZero.utils import get_external_dirs

sns.set(style="darkgrid", font_scale=1.5)
_, _, summaries_dir = get_external_dirs()

# computation parameters
number_of_blocks = 6
results_dir = Path(summaries_dir).joinpath("env3d")
results_path = Path(results_dir).joinpath(
    f"num_conf_convergence/convergence_{number_of_blocks}_blocks.pkl"
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

    df["relative energy change with final"] = (
        df["min_energy"] - _groups["min_energy"].transform("min")
    ) / _groups["min_energy"].transform("min")

    groups = df.groupby(by="smiles")
    number_of_molecules = len(groups)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(
        f"Energy distribution for a random sample of {number_of_molecules} molecules with {number_of_blocks} blocks"
    )

    list_minimum_energies = groups["min_energy"].apply(np.nanmin).values
    ax = fig.add_subplot(111)
    ax.hist(list_minimum_energies)
    ax.set_xlabel("energy (kcal/mol)")
    ax.set_ylabel("counts")
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

        ax4.plot(list_num_conf, list_relative_energy_change, "-", label=smiles)
        ax4.set_ylabel("relative change in energy (%)")

    g = df[["num_conf", "relative energy change with final"]].groupby(by="num_conf")

    for percentage in [0.0, 1.0, 2.5, 5.0, 10.]:
        list_number_converged = []
        list_num_conf = []
        for num_conf in g.groups.keys():
            group_df = g.get_group(num_conf)
            list_num_conf.append(num_conf)
            list_number_converged.append(
                np.sum(
                    group_df["relative energy change with final"] <= 0.01 * percentage
                )
            )

        list_number_converged = np.array(list_number_converged)[
            np.argsort(list_num_conf)
        ]
        ax3.plot(
            list_num_conf,
            100 * list_number_converged / number_of_molecules,
            "-",
            label=f"within {percentage} %",
        )

    ax3.set_ylabel("% converged molecules")
    ax3.legend(loc=0)

    ax3.set_xlabel("number of configuration")
    ax4.set_xlabel("number of configuration")

    fig.savefig(results_dir.joinpath(f"statistics_{number_of_blocks}_blocks.png"))
    plt.show()
