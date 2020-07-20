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
    "dataset_root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
    "file_names": ["Zinc15_260k_0",  "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
    "split_name": "ksplit_Zinc15_260k",
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
    mol = Chem.MolFromSmiles(m)
    mf = get_fp(mol,1024,[2])
    fps.append(mf)

fps = np.stack(fps,axis=0)

_, klabel = kmeans2(fps, k=20)
#print(klabel)
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

splits = [train_set, val_set, test_set]

print(len(train_set))
print(len(val_set))
print(len(test_set))

split_path = osp.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
np.save(split_path, splits)
