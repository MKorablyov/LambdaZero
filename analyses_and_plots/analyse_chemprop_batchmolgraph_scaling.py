"""
This script analyses the time scaling
"""
import logging
import time
import numpy as np
import pandas as pd
from chemprop.features import BatchMolGraph

from LambdaZero.examples.chemprop.utils import get_chemprop_graphs_from_raw_data_dataframe

import matplotlib.pyplot as plt

from analyses_and_plots import ANALYSIS_RESULTS_DIR
from analyses_and_plots.analysis_utils import get_dock_blocks105_dataframe

file_name = "dock_blocks105_walk40_clust.feather"
np.random.seed(0)

highest_power = 12
number_of_mols = 2**highest_power
number_of_trials = 20
output_path = ANALYSIS_RESULTS_DIR.joinpath("batchmolgraph_timing.png")


if __name__ == '__main__':

    raw_data_df = get_dock_blocks105_dataframe(file_name)

    list_data = get_chemprop_graphs_from_raw_data_dataframe(raw_data_df[:number_of_mols])

    list_trial_number = list(range(1, number_of_trials+1))

    columns = [f'trial {trial_number}' for trial_number in list_trial_number]
    stats_df = pd.DataFrame(columns=columns)

    for power in range(1, highest_power+1):
        logging.info(f"doing 2**{power}")
        n = 2**power

        row_dict = {}
        for column in columns:
            trial_data = np.random.choice(list_data, n, replace=False)

            assert len(trial_data) == n

            list_mols = [d['mol_graph'] for d in trial_data]
            t1 = time.time()
            _ = BatchMolGraph(list_mols)
            t2 = time.time()
            dt = t2-t1
            row_dict[column] = dt
        row_to_add = pd.Series(row_dict, name=n)
        stats_df = stats_df.append(row_to_add)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Scaling of BatchMolGraph creation with number of molecules")
    ax = fig.add_subplot(111)
    ax.set_xlabel('number of molecules to batch')
    ax.set_ylabel('BatchMolGraph creation time (seconds)')

    list_n = np.array(stats_df.index)
    list_mean = stats_df.mean(axis=1).values
    list_std = stats_df.std(axis=1).values

    ax.plot(list_n, list_mean, 'bo-', label='mean')

    list_y_min = list_mean - list_std
    list_y_max = list_mean + list_std

    ax.fill_between(list_n, list_y_min, list_y_max, alpha=0.25, label='+/- standard deviation')

    ax.legend(loc=0)

    fig.savefig(output_path)
