import json
from pathlib import Path
from typing import Dict
import pandas as pd


def process_row_for_writing_to_feather(row: Dict) -> Dict:
    """
    Convert numpy array in the dictionary to bytes array so it can be written to feather.

    Args:
        row (Dict): a row of data, containing numpy array

    Returns:
        byte_row (Dict): a row with relevant fields turned into bytes.

    """
    byte_row = dict(row)
    for key in ['coord', 'n_axis']:
        byte_row[key] = row[key].tobytes()

    return byte_row


def append_row_to_dataframe(df: pd.DataFrame, row: Dict) -> pd.DataFrame:
    return df.append(row, ignore_index=True)


def create_or_append_feather_file(feather_file_path: Path, row):

    if feather_file_path.exists():
        df = pd.read_feather(feather_file_path)
        new_df = append_row_to_dataframe(df, row)
    else:
        new_df = pd.DataFrame(data=row, index=[0])

    new_df.to_feather(str(feather_file_path))


def get_debug_blocks(block_file_path: str, number_of_blocks: int) -> Dict:
    """

    This method retains number_of_blocks in the block file for quick debugging caclulation.

    Args:
        block_file_path (str): path to original block file
        number_of_blocks (int): number of blocks to keep

    Returns:
        debug_blocks_file_path (str):  path to debug block file

    """

    with open(block_file_path, 'r') as f:
        blocks_dict = json.load(f)

        new_block_dict = dict()
        for high_level_key, values_dict in blocks_dict.items():
            sublist_of_keys = list(values_dict.keys())[:number_of_blocks]
            new_values_dict = {k: values_dict[k] for k in sublist_of_keys}
            new_block_dict[high_level_key] = new_values_dict

    return new_block_dict



