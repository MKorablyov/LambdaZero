import os.path as osp
import torch_geometric.transforms as T
import LambdaZero.inputs
import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

load_seen_config = {
    "mean":-8.6, "std": 1.1,
    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
    "dataset": LambdaZero.inputs.BrutalDock,
    "dataset_config": {
        "root": osp.join(datasets_dir, "brutal_dock/seh"),
        "props": ["dockscore", "smiles"],
        "transform": T.Compose([LambdaZero.utils.Complete()]),
        "file_names": ["Zinc20_docked_neg_randperm_3k"],
    },
}


model_config = {
    "train_epochs":75,
    "batch_size":50,
    "num_mc_samples":5,
    "device":"cuda"
}
