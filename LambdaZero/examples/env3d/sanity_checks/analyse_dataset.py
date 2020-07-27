"""
The goal of this script is to analyse a dataset.
"""
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from LambdaZero.examples.env3d.utilities import get_angles_in_degrees
from LambdaZero.utils import get_external_dirs

datasets_dir, _, summaries_dir = get_external_dirs()
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")


number_of_parent_blocks = 5

dataset_path = results_dir.joinpath(
    f"env3d_dataset_{number_of_parent_blocks}_parent_blocks.feather"
)

if __name__ == "__main__":
    df = pd.read_feather(dataset_path)

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

    plt.show()
