import os,sys, time
import os.path as osp
import numpy as np
import pandas as pd

import sys
sys.path.append("../../utils")

import LambdaZero.inputs
import LambdaZero.utils

from LambdaZero.chem import *
from rdkit import Chem

import time
from scipy.cluster.vq import kmeans2

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


DEFAULT_CONFIG = {
    "dataset_root": "/home/vbutoi/scratch/zinc15",
    "file_names": ["zinc15_full_0",
                   "zinc15_full_17",
                   "zinc15_full_24",
                   "zinc15_full_31",
                   "zinc15_full_39",
                   "zinc15_full_46",
                   "zinc15_full_53",
                   "zinc15_full_7",
                   "zinc15_full_10",
                   "zinc15_full_18",
                   "zinc15_full_25",
                   "zinc15_full_32",
                   "zinc15_full_3",
                   "zinc15_full_47",
                   "zinc15_full_54",
                   "zinc15_full_8",
                   "zinc15_full_11",
                   "zinc15_full_19",
                   "zinc15_full_26",
                   "zinc15_full_33",
                   "zinc15_full_40",
                   "zinc15_full_48",
                   "zinc15_full_55",
                   "zinc15_full_9",
                   "zinc15_full_12",
                   "zinc15_full_1",
                   "zinc15_full_27",
                   "zinc15_full_34",
                   "zinc15_full_41",
                   "zinc15_full_49",
                   "zinc15_full_56",
                   "zinc15_full_13",
                   "zinc15_full_20",
                   "zinc15_full_28",
                   "zinc15_full_35",
                   "zinc15_full_42",
                   "zinc15_full_4",
                   "zinc15_full_57",
                   "zinc15_full_14",
                   "zinc15_full_21",
                   "zinc15_full_29",
                   "zinc15_full_36",
                   "zinc15_full_43",
                   "zinc15_full_50",
                   "zinc15_full_58",
                   "zinc15_full_15",
                   "zinc15_full_22",
                   "zinc15_full_2",
                   "zinc15_full_37",
                   "zinc15_full_44",
                   "zinc15_full_51",
                   "zinc15_full_5",
                   "zinc15_full_16",
                   "zinc15_full_23",
                   "zinc15_full_30",
                   "zinc15_full_38",
                   "zinc15_full_45",
                   "zinc15_full_52",
                   "zinc15_full_6"],
    "split_name": "ksplit_total_dataset",
    "probs": [0.8, 0.1, 0.1],
}
config = DEFAULT_CONFIG

def get_fingerprint(mol):
    fp = AllChem.GetMorganFingerprint(mol)
    return fp

f = config["file_names"]
data = [pd.read_feather(osp.join(config["dataset_root"], "raw", f + ".feather")) for f in config["file_names"]]
data = pd.concat(data, axis=0)

fps = []
num_broken = 0

for m in data['smi']:
    print("Lol")
    mol = Chem.MolFromSmiles(m)
    mf = get_fp(mol,1024,[2])
    fps.append(mf)

fps = np.stack(fps,axis=0)

_, klabel = kmeans2(fps, k=20)

train_set, val_set, test_set = [],[],[]
for i in range(20):
    idx = np.where(klabel==i)[0]
    measure = np.random.uniform()
    if measure < 0.1:
        test_set.append(idx)
    elif 0.1 <= measure < 0.3:
        val_set.append(idx)
    else:
        train_set.append(idx)

train_set = np.array(np.concatenate(train_set))
val_set = np.array(np.concatenate(val_set))
test_set = np.array(np.concatenate(test_set))

train_set = np.sort(train_set[train_set < 259324])
val_set = np.sort(val_set[val_set < 259324])
test_set = np.sort(test_set[test_set < 259324])

splits = [train_set, val_set, test_set]

print(train_set)
print(val_set)
print(test_set)
print(len(train_set))
print(len(val_set))
print(len(test_set))

raise ValueError

split_path = osp.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
np.save(split_path, splits)
