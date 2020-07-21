"""
This script plots the convergence data of min_energy vs. num_conf. It assumes that the data has been generated
with another script and is written to file as a pandas pickle.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import LambdaZero

_, _, summaries_dir = LambdaZero.utils.get_external_dirs()
#results_path = Path(summaries_dir).joinpath(f'num_conf_convergence/conv_{number_of_blocks}_blocks_test.pkl')

# computation parameters
number_of_blocks = 5
results_dir = Path(summaries_dir).joinpath("env3d")
results_path = Path(results_dir).joinpath(f'num_conf_convergence/conv_{number_of_blocks}_blocks.pkl')

if __name__ == "__main__":

    df = pd.read_pickle(results_path)

    groups = df.groupby(by='smiles')

    fig = plt.figure()
    fig.suptitle(f"Energy behavior vs. number of embeddings \nfor a random sample of molecules with {number_of_blocks} blocks")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for smiles in groups.groups.keys():
        group_df = groups.get_group(smiles)
        list_num_conf = group_df['num_conf'].values
        list_min_energies = group_df['min_energy'].values
        list_z = (list_min_energies - list_min_energies.mean())/list_min_energies.std()
        ax1.plot(list_num_conf, list_z, 'o-', label=smiles)
        ax1.set_ylabel("normalized energy")

        list_dz = list_z[1:] - list_z[:-1]
        ax2.plot(list_num_conf[1:], list_dz, 'o-', label=smiles)
        ax2.set_ylabel("change in relative energy")

    ax2.set_xlabel("number of configuration")
    plt.show()
