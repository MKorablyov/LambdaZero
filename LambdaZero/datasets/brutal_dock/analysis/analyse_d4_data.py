from LambdaZero.datasets.brutal_dock import BRUTAL_DOCK_DATA_DIR
import pandas as pd
import matplotlib.pyplot as plt

file_name = 'dock_blocks105_walk40_clust.feather'

d4_feather_data_path = BRUTAL_DOCK_DATA_DIR.joinpath(f"d4/raw/{file_name}")

if __name__ == '__main__':

    df = pd.read_feather(d4_feather_data_path)

    fig = plt.figure(figsize=(12, 8))


    fig.suptitle(f'Distribution of scores in {file_name}')
    ax = fig.add_subplot(111)
    df['gridscore'].hist(ax=ax, bins=100)

    ax.set_xlabel('gridscore')
    ax.set_ylabel('counts')
