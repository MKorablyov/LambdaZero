import os
import numpy as np
import os.path as osp
import torch as th
import torch.nn.functional as F
from torch.utils.data import Subset

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
        "targets": ["gridscore"],
        "target_norms": [[-26.3, 12.3]],
        "file_names": ["zinc15_full_2.feather", "zinc15_full_46.feather", "zinc15_full_13.feather",
                       "zinc15_full_31.feather", "zinc15_full_47.feather", "zinc15_full_16.feather",
                       "zinc15_full_33.feather", "zinc15_full_48.feather", "zinc15_full_26.feather",
                       "zinc15_full_38.feather", "zinc15_full_54.feather", "zinc15_full_28.feather",
                       "zinc15_full_40.feather"],
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

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

model = LambdaZero.models.MPNNet()
model.load_state_dict(th.load('/home/vbutoi/LambdaZero/summaries/predictions/vanilla_mpnn_L2/'
                              'BasicRegressor_85ff6e04_0_2020-07-14_00-14-34rwsgu2d5/checkpoint_80/model.pth'))
model.to(device)
model.eval()

config = config['trainer_config']

target = config["targets"][0]
target_norm = config["target_norms"][0]

dataset = LambdaZero.inputs.BrutalDock(config["dataset_root"],
                                        props=config["molprops"],
                                        transform=config["transform"],
                                        file_names=config["file_names"])
# split dataset
bsize = 1

val_subset = dataset
loader = DL(val_subset, batch_size=bsize)

mol_checkpoints = []
regrets = []
med_regrets = []

running_preds_true_energy = th.tensor([],requires_grad=False)
running_preds = th.tensor([],requires_grad=False)
running_gc = th.tensor([],requires_grad=False)

num_mol_seen = 0

real_energies = []
predictions = []

for bidx, data in enumerate(loader):

    data = data.to(device)

    with th.no_grad():
        logit = model(data)

    pred = (logit * target_norm[1]) + target_norm[0]

    real_energies.append(data.gridscore.item())
    predictions.append(pred.item())

    num_mol_seen += bsize

    y = getattr(data, target)

    cpu_y = y.cpu()
    cpu_pred = pred.cpu()

    highest_ys = th.topk(y.cpu(), config['num_rank'])
    running_gc = th.cat((running_gc, highest_ys[0]), 0)
    running_gc = th.topk(running_gc, config['num_rank'])[0]

    highest_preds = th.topk(cpu_pred, config['num_rank'])

    running_preds = th.cat((running_preds, highest_preds[0]), 0)
    running_preds_true_energy = th.cat((running_preds_true_energy, cpu_y[highest_preds[1]]), 0)

    running_preds = th.topk(running_preds, config['num_rank'])

    running_preds_true_energy = running_preds_true_energy[running_preds[1]].detach()
    running_preds = running_preds[0].detach()

    median_regret = abs(np.median(running_preds_true_energy) - np.median(running_gc)).item()
    regret = F.l1_loss(running_preds_true_energy, running_gc).item()

    regrets.append(regret)
    mol_checkpoints.append(num_mol_seen)
    med_regrets.append(median_regret)

plt.title("Predictions vs Real")
plt.xlabel("Real Energies")
plt.ylabel("Predictions")
plt.scatter(real_energies, predictions)
plt.savefig("/home/vbutoi/LambdaZero/summaries/predictions/vanilla_mpnn_L2/pred_fig_1.png")

plt.title("Regret Over Time")
plt.xlabel("Num Mols Seen")
plt.ylabel("MSE Regret")
plt.plot(num_mol_seen, regrets, label="mae regret")
plt.plot(num_mol_seen, med_regrets, label="median regret")
plt.savefig("/home/vbutoi/LambdaZero/summaries/predictions/vanilla_mpnn_L2/pred_fig_1.png")

"""
    num_mol_seen += bsize

    pred = (logit * target_norm[1]) + target_norm[0]
    y = getattr(data, target)

    cpu_y = y.cpu()
    cpu_pred = pred.cpu()

    highest_ys = th.topk(y.cpu(), config['num_rank'])
    running_gc = th.cat((running_gc, highest_ys[0]), 0)
    running_gc = th.topk(running_gc, config['num_rank'])[0]

    highest_preds = th.topk(cpu_pred, config['num_rank'])

    running_preds = th.cat((running_preds, highest_preds[0]), 0)
    running_preds_true_energy = th.cat((running_preds_true_energy, cpu_y[highest_preds[1]]), 0)

    running_preds = th.topk(running_preds, config['num_rank'])

    running_preds_true_energy = running_preds_true_energy[running_preds[1]].detach()
    running_preds = running_preds[0].detach()

    median_regret = abs(np.median(running_preds_true_energy) - np.median(running_gc)).item()
    regret = F.l1_loss(running_preds_true_energy, running_gc).item()
    mse_regret = F.mse_loss(running_preds_true_energy, running_gc).item()

    regrets.append(regret)
    mol_checkpoints.append(num_mol_seen)
"""