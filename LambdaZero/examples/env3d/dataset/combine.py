"""
This script combines multiple feather files containing the data into a single feather file.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Combine feather files into a single file"
    )

    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="directory containing the input feather files",
    )

    args = parser.parse_args(sys.argv[1:])
    data_path = Path(args.data_path)
    output_path = data_path.joinpath("combined/combined_dataset.feather")
    output_path.parent.mkdir(exist_ok=True)

    list_df = []
    for file_path in data_path.glob("*.feather"):
        list_df.append(pd.read_feather(file_path))

    df = pd.concat(list_df).reset_index(drop=True)

    # This column renaming is necessary because the code initially used
    # some column names ending in *_index; However, the method
    #           torch_geometric.data.data.Data.__inc__
    # treats variable names containing "index" in a special way, which
    # produces unexpected effects when using a dataloader and batching.
    # To avoid this issue, we rename here.
    df.rename(columns={"attachment_node_index": "attachment_node_idx",
                       "attachment_block_index": "attachment_block_class"},
              inplace=True)

    df.to_feather(output_path)
