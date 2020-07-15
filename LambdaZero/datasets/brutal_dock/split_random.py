import os,sys, time
import os.path as osp
import numpy as np
import pandas as pd

import LambdaZero.inputs
import LambdaZero.utils

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


DEFAULT_CONFIG = {
    #"dataset_root": os.path.join(datasets_dir, "brutal_dock/d4"),
    #"file_names": ["dock_blocks105_walk40_clust"],
    #"split_name": "randsplit_dock_blocks105_walk40_clust",

    "dataset_root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
    "file_names": ["Zinc15_260k_0",  "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
    "split_name": "randsplit_Zinc15_260k",
    "probs": [0.8, 0.1, 0.1],
}
config = DEFAULT_CONFIG


if __name__ == "__main__":

    f = config["file_names"]
    data = [pd.read_feather(osp.join(config["dataset_root"], "raw", f + ".feather")) for f in config["file_names"]]
    data = pd.concat(data, axis=0)

    from matplotlib import pyplot as plt
    score = data["gridscore"].to_numpy()

    plt.show()
    print(data["gridscore"].to_numpy().std())


    data.reset_index(drop=True, inplace=True)
    splits = LambdaZero.inputs.random_split(len(data), config["probs"])

    split_path = osp.join(config["dataset_root"], "raw", config["split_name"] + ".npy")
    np.save(split_path, splits)

    train_set, val_set, test_set = np.load(split_path, allow_pickle=True)