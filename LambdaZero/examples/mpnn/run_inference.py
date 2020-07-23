import os
import numpy as np
import os.path as osp
import torch as th
import torch.nn.functional as F
from torch.utils.data import Subset

import ray

from LambdaZero.utils import get_external_dirs
import LambdaZero.inputs
import LambdaZero.models

from custom_dataloader import DL

from matplotlib import pyplot as plt

transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

config = {
    "trainer_config": {
        "dataset_root": "/home/vbutoi/scratch/zinc_data",
        "targets": "gridscore",
        "target_norm": [-43.042, 7.057],
        "file_names": ["zinc15_full_2", "zinc15_full_46", "zinc15_full_13",
                       "zinc15_full_31", "zinc15_full_47", "zinc15_full_16"],
                       """
                       "zinc15_full_33", "zinc15_full_48", "zinc15_full_26",
                       "zinc15_full_38", "zinc15_full_54", "zinc15_full_28",
                       "zinc15_full_40"],
                       """
        "transform": transform,
        "lr": 0.001,
        "b_size": 64,
        "dim": 64,
        "num_epochs": 5,
        "num_rank": 15,
        "molprops": ["gridscore"],
        "model": "mpnn"
        }
}

ray.init()

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

model = LambdaZero.models.MPNNet()
model.load_state_dict(th.load('/home/vbutoi/LambdaZero/summaries/knn_split/mpnn000_L2/BasicRegressor_26b1482e_0_2020-07-21_02-05-17y98a028f/checkpoint_100/model.pth'))
model.to(device)
model.eval()

config = config['trainer_config']

target = config["targets"][0]
target_norm = config["target_norm"]

dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                        props=config["molprops"],
                                        transform=LambdaZero.utils.Complete(),
                                        file_names=config["file_names"])
# split dataset
bsize = 128

val_subset = dataset
loader = DL(val_subset, batch_size=bsize)

normalizer = LambdaZero.utils.MeanVarianceNormalizer(config["target_norm"])
num_mol_seen = 0

epoch_targets = []
epoch_preds = []

top_fifteen = []
top_fifty = []
mol_seen = []

print("About to start")
i = 0

for bidx, data in enumerate(loader):
    
    print("Current iter: " + str(i))
    i += 1
    data = data.to(device)
    targets = getattr(data, config["targets"])

    with th.no_grad():
        logits = model(data)

    num_mol_seen += bsize
    epoch_targets.append(targets.detach().cpu().numpy())
    epoch_preds.append(normalizer.unnormalize(logits).detach().cpu().numpy())

    e_targ = np.concatenate(epoch_targets, 0)
    e_pred = np.concatenate(epoch_preds, 0)

    ranked_targets = e_targ[np.argsort(e_targ)]
    predsranked_targets = e_pred[np.argsort(epoch_preds)]

    top_fifteen.append(np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15]))
    top_fifty.append(np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50]))
    mol_seen.append(num_mol_seen)

plt.title("Num Mols Seen vs Top 15 Med Regret")
plt.xlabel("Num Mols Seen")
plt.ylabel("Regret")
plt.scatter(mol_seen, top_fifteen)
plt.savefig("/home/vbutoi/scratch/charts/regret_over_time15.png")

plt.title("Num Mols Seen vs Top 50 Med Regret")
plt.xlabel("Num Mols Seen")
plt.ylabel("Regret")
plt.plot(mol_seen, top_fifty)
plt.savefig("/home/vbutoi/scratch/charts/regret_over_time50.png")
