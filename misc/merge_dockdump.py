import pandas as pd
import argparse
import numpy as np
import sqlite3
import os
from pandas.io import sql
from copy import deepcopy
import gc
import os, psutil
from datetime import datetime
import glob
import time
from typing import List, Tuple
from datetime import datetime
import re

from LambdaZero.utils import get_external_dirs
from LambdaZero.contrib.oracle.oracle import PreDockingDB


datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def merge_data(data_path: str = None, test_duplicates: int = 100, duplicates_th: float = 0.2):
    """
        Get data from docking dump files and add them to the previous database.
        DB is indexed by smiles and has 1 float16 column for dockscores.
        The other columns are strings - and get be converted with eval().
    """
    if data_path is None:
        data_path = os.path.join(summaries_dir, "docking_dump")

    data_files = glob.glob(f"{data_path}/data_*.log")

    assert len(data_files) > 0, f"No data files to add to db from: {data_path}"

    all_data = []
    for i, data_f in enumerate(data_files):
        print(f"Reading from {data_f} ({i}/{len(data_files)}) ...")

        with open(data_f, "r") as f:
            lines = f.readlines()
            columns = eval(lines[0])
            # Some data was saved from np.array
            data = [eval(x.replace("array", "list")) for x in lines[1:]]
            df = pd.DataFrame(data, columns=columns)
            all_data.append(df)

    df = pd.concat(all_data)
    df.set_index('smiles', inplace=True)

    print(f"Got data for {len(df)} molecules ({len(np.unique(df.index))} unique)")

    # Cannot store columns of new size
    for x in columns[2:]:
        df[x] = df[x].apply(str)
    df["dockscore"] = df["dockscore"].astype("float16")

    # Load previous DB
    best_db, _ = PreDockingDB.get_last_db()
    if best_db is not None:
        print(f"Load previous database {best_db} ...")
        store = pd.HDFStore(best_db)
        prev_df = store.select('df')

        # Check if too many duplicates
        if test_duplicates > 0 and duplicates_th < 1.:
            df_index = df.index
            prev_df_index = prev_df.index
            test_cnt = min(len(df_index), test_duplicates)
            test_keys = np.random.choice(df_index, (test_cnt,), replace=False)
            has_keys = sum([1 if x in prev_df_index else 0 for x in test_keys])
            if has_keys/test_cnt > duplicates_th:
                r = input(f"Found {has_keys/test_cnt*100}% duplicates out of {test_cnt} samples."
                          f" Continue (y/n) ?")
                if r.lower() != "y":
                    exit()

        print(f"Ready to merge. Adding {len(df)} new rows to previous {len(prev_df)} rows ...")
        df = pd.concat([prev_df, df])
        print("Merged!")

    # Write fixed HDF5 db
    new_db_path = PreDockingDB.get_new_db_name()
    print(f"Write new db @ {new_db_path} ... ")
    df.to_hdf(new_db_path, 'df', mode='w')
    print(f"Writen!")

    r = input(f"Is the merge correct and new db has been writen to disk successfully?"
              f" Delete merged files (y/n) ?")
    if r.lower() == "y":
        print("Remove files ...")
        for d_file in data_files:
            os.remove(d_file)
    else:
        print("Exit without deleting files!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Change dir to load docking data logs (default: summaries_dir/docking_dump/)'
    )
    parser.add_argument('--test-duplicates', type=int, default=100,
                        help='Check if data was already added to db (number of molecules to check)')
    parser.add_argument('--duplicates-th', type=float, default=0.2,
                        help='Raise possible data duplicates issue above this threshold')
    args = parser.parse_args()

    merge_data(data_path=args.data_path, test_duplicates=args.test_duplicates,
               duplicates_th=args.duplicates_th)
