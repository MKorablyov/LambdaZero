import argparse
import os
import pandas as pd

from LambdaZero.datasets.postera.docking import DockVina_smi, GenMolFile_v1
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, _ = get_external_dirs()
dataset_path = os.path.join(datasets_dir, "zinc20", "zinc20_smiles_from_2D.csv")

config = {
        "outpath": os.path.join(datasets_dir, "zinc20", "docked_molecules"),
        "smina_bin": os.path.join(programs_dir, "smina", "smina.static"),
        "docksetup_dir": os.path.join(datasets_dir, "seh", "4jnc"),
        "dock_pars": "--cpu 1",
        "gen_molfile": GenMolFile_v1,
        "gen_molfile_par": {
            "outpath": os.path.join(datasets_dir, "zinc20", "input_molecules"),
            "num_conf": 20,
        }
}

dock_smi = DockVina_smi(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, help='Number of molecules in batch')
    parser.add_argument('batch_idx', type=int, help='Index of the batch')
    args = parser.parse_args()
    batch_size, batch_idx = args.batch_size, args.batch_idx

    col_names = pd.read_csv(dataset_path, nrows=0).columns.tolist()                                         # smiles, int_zinc_id
    data = pd.read_csv(dataset_path, names=col_names, nrows=batch_size, skiprows=1+batch_idx*batch_size)    # additional 1 skips column names

    for smi, idx in zip(data['smiles'], data['int_zinc_id']):
        dock_smi.dock(smi, str(idx))
