import os, sys, time
import numpy as np
import pandas as pd
sys.path.append("../../")
from affinity_torch.py_tools import chem
from affinity_torch.py_tools import multithread as mtr

class cfg:
    # temp
    db_path = "/home/maksym/Datasets/brutal_dock"
    docked_index = "blocks105_walk40_100k.parquet"

    # docking parameters
    dockout_path = "/home/maksym/Desktop/model_summaries/dock_blocks105_walk40"
    dock6_dir = "/home/maksym/Programs/dock6"
    chimera_dir = "/home/maksym/Programs/chimera"
    docksetup_dir = "/home/maksym/Datasets/brutal_dock/d4/docksetup"

if __name__ == "__main__":
    chunk_size = 8
    dock_index = pd.read_parquet(os.path.join(cfg.db_path,cfg.docked_index))
    idxs = np.arange(len(dock_index))
    np.random.shuffle(idxs)
    num_chunks = len(dock_index) // chunk_size
    dock_smi = chem.Dock_smi(outpath=cfg.dockout_path,
                             chimera_dir=cfg.chimera_dir,
                             dock6_dir=cfg.dock6_dir,
                             docksetup_dir=cfg.docksetup_dir)

    for chunk_id in range(num_chunks-1):
        dock_index_chunk = dock_index.iloc[chunk_size * chunk_id:chunk_size * (chunk_id+1)]
        args,outs,errs = mtr.run(dock_smi.dock_smi,dock_index_chunk, cols={"smi":"smiles"},tabular=True, num_threads=8)
        docked_chunk = pd.merge(args, outs, left_index=True, right_index=True)
        docked_chunk.to_parquet(os.path.join(cfg.dockout_path, "docked_" + str(chunk_id) + ".feather"))