import os.path as osp
import numpy as np
import pandas as pd
import LambdaZero.utils

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = {

    "dataset_root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
    "file_names": ["Zinc15_260k_0",  "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
    "subsample_name": "Zinc15_2k",
    "num_samples": 2000,
}

config = DEFAULT_CONFIG

if __name__ == "__main__":
    f = config["file_names"]
    data = [pd.read_feather(osp.join(config["dataset_root"], "raw", f + ".feather")) for f in config["file_names"]]
    data = pd.concat(data, axis=0, ignore_index=True)

    sel_idxs = np.arange(data.shape[0])
    np.random.shuffle(sel_idxs)
    sel_idxs = sel_idxs[:config["num_samples"]]

    sel_data = data.iloc[sel_idxs]
    sel_data = sel_data.reset_index()
    sel_data.to_feather(osp.join(config["dataset_root"], "raw", config["subsample_name"]) + ".feather")
