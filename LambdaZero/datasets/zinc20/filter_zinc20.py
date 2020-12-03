import os
import numpy as np
import pandas as pd

from LambdaZero.utils import get_external_dirs
datasets_dir, _, _ = get_external_dirs()
zinc20_root = os.path.join(datasets_dir, "zinc20")


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(zinc20_root, "zinc20_base.csv"))[['int_zinc_id', 'smiles', 'has_B', 'has_Sn']]
    data.sort_values('int_zinc_id', inplace=True)

    # determine entries with the same smiles, mark the ones with smaller int_zinc_id as duplicates and set aside
    duplicated_smiles = data[data.duplicated(subset='smiles', keep='last')].reset_index(drop=True)[['int_zinc_id', 'smiles']]

    # remove entries with duplicated smiles
    data_unique = data.drop_duplicates(subset='smiles', keep='last').reset_index(drop=True)

    # B and Sn are not supported by MMFF94 - filter out
    data_filtered = data_unique[np.logical_and(data_unique.has_B == 0, data_unique.has_Sn == 0)][['int_zinc_id', 'smiles']]

    # randomly shuffle, so that molecule generation and docking time is roughly uniformly distributed
    data_filtered = data_filtered.sample(frac=1, random_state=42)

    # following files contain only int_zinc_id and smiles
    # for additional properties merge with zinc20_base over int_zinc_id
    duplicated_smiles.to_csv(os.path.join(zinc20_root, "zinc20_duplicated_smiles.csv"), index=False)
    data_filtered.to_csv(os.path.join(zinc20_root, "zinc20_filtered.csv"), index=False)
    print(len(data_filtered))
