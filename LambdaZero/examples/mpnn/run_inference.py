import numpy as np
import torch
import sys
import os
from LambdaZero.utils import get_external_dirs, BasicRegressor
from torch_geometric.data import DataLoader
import LambdaZero.inputs
import LambdaZero.models
from LambdaZero.examples.mpnn import config
import LambdaZero.utils

transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "mpnn000_L1"

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "target": "gridscore",
        "target_norm": [-43.042, 7.057],
        "dataset_split_path": "/home/vbutoi/scratch/zinc15/raw/randsplit_sevenmil.npy",
        "b_size": 32,

        "dataset": LambdaZero.inputs.BrutalDock,
        "dataset_config": {
            "root": "/home/vbutoi/scratch/zinc15/",
            "props": ["gridscore"],
            "transform": transform,
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
                           "zinc15_full_6"]
        },

        "model": LambdaZero.models.MPNNet,
        "model_config": {},

        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },

        "train_epoch": None,
        "eval_epoch": None,
    },

    "summaries_dir": summaries_dir,
    "memory": 10 * 10 ** 9,

    "stop": {"training_iteration": 200},
    "resources_per_trial": {
        "cpu": 4,  # fixme - calling ray.remote would request resources outside of tune allocation
        "gpu": 1.0
    },
    "keep_checkpoint_num": 2,
    "checkpoint_score_attr": "train_loss",
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 100,
}
root_dir = '/home/vbutoi/LambdaZero/summaries/basic/MPNN-BASIC/' + config_name + '/'
regressor_file = None
for f in os.listdir(root_dir):
    if "BasicRegressor" in f:
        regressor_file = f

model_file = root_dir + regressor_file + '/checkpoint_100/model.pth'
config = DEFAULT_CONFIG['trainer_config']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = config["dataset"](**config["dataset_config"])
model = config["model"](**config["model_config"])
model.load_state_dict(torch.load(model_file))
model.to(device)
model.eval()
# split dataset
loader = DataLoader(dataset, batch_size=512)
normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
model.eval()
metrics = {"loss":0,}
epoch_targets = []
epoch_preds = []

i = 0 
for bidx,data in enumerate(loader):
    i+=1
    print(i)
    data = data.to(device)
    targets = getattr(data, config["target"])
    with torch.no_grad():
        logits = model(data)

    epoch_targets.append(targets.detach().cpu().numpy())
    epoch_preds.append(normalizer.itfm(logits).detach().cpu().numpy())

epoch_targets = np.concatenate(epoch_targets,0)
epoch_preds = np.concatenate(epoch_preds, 0)

ranked_targets = epoch_targets[np.argsort(epoch_targets)]
predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])

np.save(root_dir + 'top25',np.array(metrics["top15_regret"]))
np.save(root_dir + 'top50',np.array(metrics["top50_regret"]))
