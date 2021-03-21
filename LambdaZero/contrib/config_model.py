import os.path as osp
import torch_geometric.transforms as T
import LambdaZero.inputs
import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
import LambdaZero.utils

load_seen_config = {
    "mean": -8.6, "std": 1.1,
    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
    "raw_path": osp.join(datasets_dir, "brutal_dock/seh/raw"),
    "proc_path": osp.join(datasets_dir, "brutal_dock/seh/processed_rlbo"),
    "file_names": ["Zinc20_docked_neg_randperm_3k"],
}


model_config = {
    "train_epochs":75,
    "batch_size":75,
    "mpnn_config":{
        "drop_last":True,
        "drop_data":False,
        "drop_weights":True,
        "drop_prob":0.1,
        "num_feat":14
    },
    "lr":1e-3,
    "transform":None,
    "num_mc_samples":10,
    "device":"cuda"
}
