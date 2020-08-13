"""
This script is used to extract a small subset of the full dataset where
each block class is represented according to its frequency in the dataset,
with at least one entry for each block.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from LambdaZero.utils import get_external_dirs

_, _, summaries_dir = get_external_dirs()

#  Idiosyncratic path on my local machine. Change as needed.
raw_data_path = Path(summaries_dir).joinpath("env3d/dataset/from_cluster/RUN5/data/raw/")


input_data_path = raw_data_path.joinpath("combined_dataset.feather")
output_data_path = raw_data_path.joinpath("small_dataset.feather")

number_of_samples = 1000

if __name__ == '__main__':

    np.random.seed(0)

    df = pd.read_feather(input_data_path)

    normalized_value_counts = df['attachment_block_class'].value_counts(normalize=True)
    number_of_samples_per_class = np.ceil(number_of_samples*normalized_value_counts)

    groups = df.groupby(by='attachment_block_class')

    list_df = []
    for class_index, group_df in groups:
        number_of_samples = int(number_of_samples_per_class.loc[class_index])
        list_df.append(group_df[:number_of_samples])

    sample_df = pd.concat(list_df).reset_index(drop=True)

    sample_df.to_feather(output_data_path)
