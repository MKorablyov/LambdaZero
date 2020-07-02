import os
import socket
from pathlib import Path

import numpy as np
import torch

from LambdaZero.utils import get_external_dirs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class cfg:
    machine = socket.gethostname()
    if machine == "Ikarus":
        datasets_dir = "/home/maksym/Datasets"
        programs_dir = "/home/maksym/Programs"
        summaries_dir = "/home/maksym/Desktop/model_summaries"
    else:
        datasets_dir, programs_dir, summaries_dir = get_external_dirs()

    load_model = None  # os.path.join(datasets_dir, "brutal_dock/d4/d4_100k_mine_model_001")
    db_root = os.path.join(datasets_dir, "brutal_dock/d4")
    db_name = "dock_blocks105_walk40_12_clust"
    molprops = ["gridscore", "klabel"]
    target = "gridscore"
    file_names = ["dock_blocks105_walk40_clust", "dock_blocks105_walk40_2_clust"]  # fixme ifcoord in molmdp
    target_norm = [-26.3, 12.3]
    test_prob = 0.1
    valid_prob = 0.1
    b_size = 16
    dim = 64
    num_epochs = 120
    outpath = str(Path(summaries_dir).joinpath("brutal_dock"))
    model_name = db_name + "model002"


def random_split(dataset, test_prob, valid_prob,
                  test_idx=torch.tensor([], dtype=torch.long),
                  train_idx=torch.tensor([], dtype=torch.long),
                  val_idx=torch.tensor([], dtype=torch.long)):

    num_last_split = (len(test_idx) + len(train_idx) + len(val_idx))
    num_split = len(dataset) - num_last_split

    ntest = int(num_split * test_prob)
    nvalid = int(num_split * valid_prob)
    idx = torch.randperm(num_split) + num_last_split

    test_idx = torch.cat([test_idx, idx[:ntest]])
    val_idx = torch.cat([val_idx, idx[ntest:ntest + nvalid]])
    train_idx = torch.cat([train_idx, idx[ntest + nvalid:]])

    return test_idx, val_idx, train_idx


def knn_split(klabels, probs):
    "split the dataset based on the KNN cluster assignment in some space"
    klabels = np.asarray(klabels)
    nsplits = len(probs)
    assert len(klabels.shape) == 1, "bad input shape for split"
    assert np.isclose(np.sum(probs), 1.0), "probabilities do not sum up to 1"

    klabel_sort = np.argsort(klabels)
    klabel_unq, klabel_count = np.unique(klabels, return_counts=True)
    klabel_slices = np.concatenate([np.asarray([0]), np.cumsum(klabel_count)], 0)

    splits = [[] for _ in range(nsplits)]
    for i, klabel in enumerate(klabel_unq):
        klabel_idx = klabel_sort[klabel_slices[i]:klabel_slices[i + 1]]
        draw = np.random.choice(np.arange(nsplits), 1, p=probs, replace=True)[0]
        splits[draw].append(klabel_idx)
    splits = (np.concatenate(sp, 0) for sp in splits)
    return splits


