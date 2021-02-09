import time
import os.path as osp
import numpy as np


def load_data_v1(target, dataset_split_path, dataset, dataset_config):
    # make dataset
    dataset = dataset(**dataset_config)
    train_idxs, val_idxs, test_idxs = np.load(dataset_split_path, allow_pickle=True)
    # take corresponding indices from data loaders
    train_x = [dataset[int(i)] for i in train_idxs]
    train_y = [getattr(dataset[int(i)],target) for i in train_idxs]
    val_x = [dataset[int(i)] for i in val_idxs]
    val_y = [getattr(dataset[int(i)],target) for i in val_idxs]
    return train_x, train_y, val_x, val_y




#load_dataset_v1(data_config)