import sys, os, time
import numpy as np
import os.path as osp
import pandas as pd
from matplotlib import pyplot as plt

import LambdaZero.utils

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

# save to the datasets
docked = osp.join(summaries_dir, "save_docked")
docked = [pd.read_feather(osp.join(docked, f)) for f in os.listdir(docked)]
docked = pd.concat(docked, axis=0, ignore_index=True)
docked.to_feather(osp.join(datasets_dir, "brutal_dock/seh/raw/random_molecule_proxy_20k.feather"))
