import os,sys, time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from scipy.cluster.vq import kmeans2
from affinity_torch.py_tools import chem
from affinity_torch.py_tools import multithread as mtr
from affinity_torch.py_tools import chem


# read smiles
# make fingerprints
# make k-means
class cfg:
    db_path = "/home/maksym/Datasets/brutal_dock/d4/raw"
    docked_index = "dock_blocks105_walk40_2.feather"
    docked_index_clust = "dock_blocks105_walk40_2_clust.feather"
    #docked_index = "d4_0k_mine.parquet"
    #docked_index_clust = "dock_blocks105_walk40_clust.feather"

if __name__ == "__main__":
    # compute embeddings
    docked_idx = pd.read_feather(os.path.join(cfg.db_path, cfg.docked_index))
    args, outs, errs = mtr.run(chem.get_fingerprint, docked_idx, cols={"smiles": "smi"}, debug=True, tabular=True)
    docked_idx = pd.merge(args, outs, left_index=True, right_index=True)
    fingerprints = np.stack(docked_idx["fingerprint"].to_list(), 0)

    # compute knn
    start = time.time()
    _, klabel = kmeans2(fingerprints, k=200)
    klabel = pd.DataFrame({"klabel": klabel})
    docked_idx = pd.merge(docked_idx, klabel, left_index=True, right_index=True)
    docked_idx = docked_idx.drop(columns=["fingerprint"]) # drop heavy to save fingerprint

    # docked_idx = docked_idx.rename(columns={"smi":"smiles"})
    docked_idx.to_feather(os.path.join(cfg.db_path, cfg.docked_index_clust))
    docked_idx = pd.read_feather(os.path.join(cfg.db_path, cfg.docked_index_clust))
    print(docked_idx)
    #print(docked_idx[0])