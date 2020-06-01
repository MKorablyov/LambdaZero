import matplotlib.pyplot as plt

from analyses_and_plots.analysis_utils import get_dock_blocks105_dataframe

file_name = "dock_blocks105_walk40_clust.feather"

if __name__ == '__main__':

    df = get_dock_blocks105_dataframe(file_name)

    fig = plt.figure(figsize=(12, 8))

    fig.suptitle(f'Distribution of scores in {file_name}')
    ax = fig.add_subplot(111)
    df['gridscore'].hist(ax=ax, bins=100)

    ax.set_xlabel('gridscore')
    ax.set_ylabel('counts')
