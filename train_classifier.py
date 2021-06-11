"""
Training a Classifier
"""
import copy

import torch
import numpy as np
import torch.optim as optim
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import wandb
from argparse import Namespace
import os
from copy import deepcopy
import itertools
import sys
import time
import functools

from lightrl.utils.utils import set_seed
from lightrl.utils.utils import setup_loggers
from lightrl.reg_models import get_actor_model
from lightrl.env.vec_env import fast_from_data_list
from lightrl.utils.utils import SummaryStats
from lightrl.utils.radam_optimizer import RAdam

datasets_dir, programs_dir, summaries_dir = None, None, None


class CustomDataset(Dataset):
    def __init__(self, graphs, targets, smis, bins, classification=True, return_extra=False):
        self._graphs = graphs
        self._smis = smis
        self._bins = bins
        self._return_extra = return_extra

        self._target_val = targets
        self.classification = classification

        if classification:
            targets = torch.argmax(targets, dim=1)
        else:
            targets = targets

        self._targets = targets

    def __getitem__(self, idx):
        sample = {'graph': self._graphs[idx],
                  'target': self._targets[idx],
                  "smi": self._smis[idx]
                  }
        if self.classification and self._return_extra:
            sample["target_v"] = self._target_val[idx]
            sample["bins"] = self._bins[idx]

        return sample

    def __len__(self):
        return len(self._graphs)


def collate_fn(data, device):
    """ Used in dataloader to merge batch """
    _graphs = fast_from_data_list([x["graph"] for x in data]).to(device)
    _graphs["smiles"] = [x["smi"] for x in data]
    _tgt = torch.stack([x["target"] for x in data]).to(device)
    if "target_v" in data[0]:
        _graphs["target_v"] = torch.stack([x["target_v"] for x in data]).to(device)

    if "bins" in data[0]:
        _graphs["bins"] = torch.stack([x["bins"] for x in data]).to(device)

    return _graphs, _tgt


def train_epoch(ep, loader, model, optimizer, criterion, device, classification=True):
    model.train()

    correct = 0
    total = 0

    running_loss = 0.0

    do_res = (not classification) and (ep + 1) % 50 == 0
    res = dict() if not do_res else dict({k: [] for k in [3, 5, 10, 15]})

    for bidx, (data_graph, data_tgt) in enumerate(loader):
        data_graph, data_tgt = data_graph.to(device), data_tgt.to(device)
        optimizer.zero_grad()

        _, y_hat = model(data_graph)
        loss = criterion(y_hat, data_tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if not classification:
            if do_res:
                # rank
                tgt_sort = torch.argsort(data_tgt, axis=1)
                pred_sort = torch.argsort(y_hat.data, axis=1)
                for topk in res.keys():
                    for irow in range(len(tgt_sort)):
                        tgt_set = set(tgt_sort[irow][:topk].data.cpu().numpy())
                        pred_set = set(pred_sort[irow][:topk].data.cpu().numpy())
                        match = len(set.intersection(tgt_set, pred_set)) /float(topk)
                        res[topk].append(match)

            data_tgt = torch.argmax(data_tgt, dim=1)

        _, predicted = torch.max(y_hat.data, 1)
        correct += (predicted == data_tgt).sum().item()

        total += data_tgt.size(0)

    if do_res:
        for topk in list(res.keys()):
            res[str(topk)] = np.mean(res.pop(topk))

    return correct / total, running_loss / total, res


def eval_epoch(loader, model, criterion, device, classification=True):
    model.eval()

    correct = 0
    total = 0

    topks = [1, 3, 5, 10, 20, 50]
    res = dict()
    for topk in topks:
        res[f"fmatch_top-{topk}"] = []
        res[f"top-{topk}"] = 0
        res[f"inv_top-{topk}"] = 0

    running_loss = 0.0
    with torch.no_grad():
        for bidx, (data_graph, data_tgt) in enumerate(loader):
            data_graph, data_tgt = data_graph.to(device), data_tgt.to(device)
            _, y_hat = model(data_graph)
            loss = criterion(y_hat, data_tgt)
            running_loss += loss.item()

            if classification:
                data_tgt_v = data_graph["target_v"]
            else:
                data_tgt_v = data_tgt_v

            # ======================================================================================
            # Calculate topk match
            tgt_sort = torch.argsort(data_tgt_v, axis=1, descending=True)
            pred_sort = torch.argsort(y_hat.data, axis=1, descending=True)

            # Go to bins groupping
            tgt_sort = data_graph.bins.gather(1, tgt_sort).data.cpu().numpy()
            pred_sort = data_graph.bins.gather(1, pred_sort).data.cpu().numpy()

            for topk in topks:
                top_target = tgt_sort[:, :topk]
                top_pred = pred_sort[:, :topk]
                for irow in range(len(tgt_sort)):
                    tgt_set = set(top_target[irow])
                    pred_set = set(top_pred[irow])
                    match = len(set.intersection(tgt_set, pred_set)) / len(set.union(tgt_set, pred_set))
                    res[f"fmatch_top-{topk}"].append(match)

                res[f"top-{topk}"] += ((top_pred - tgt_sort[:, 0][:, None]) == 0).any(axis=1).sum()
                res[f"inv_top-{topk}"] += ((top_target - pred_sort[:, 0][:, None]) == 0).any(axis=1).sum()

            if not classification:
                data_tgt = torch.argmax(data_tgt, dim=1)

            # ======================================================================================
            _, predicted = torch.max(y_hat.data, 1)
            total += data_tgt.size(0)
            correct += (predicted == data_tgt).sum().item()

    for topk in topks:
        res[f"fmatch_top-{topk}"] = np.mean(res[f"fmatch_top-{topk}"])
        res[f"top-{topk}"] = res[f"top-{topk}"] / total
        res[f"inv_top-{topk}"] = res[f"inv_top-{topk}"] / total

    return correct / total, running_loss / total, res


def get_data(dataset_filename, device, cfg, env=None, ignore_smiles=[]):
    from LambdaZero.environments import BlockMolEnvGraph_v1
    import LambdaZero

    datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

    binning_decimal = getattr(cfg, "binning_decimal", 1)
    dataset_filter_candidates = getattr(cfg, "dataset_filter_candidates", True)
    dataset_score_candidates = getattr(cfg, "dataset_score_candidates", False)
    qed_th = getattr(cfg, "qed_th", 0.3)
    synth_th = getattr(cfg, "synth_th", 4.)

    # Load env_data
    env_data = np.load(f"{datasets_dir}/{dataset_filename}", allow_pickle=True).item()

    print(f"[LOAD DATASET] Loaded data for {len(env_data)} states (ignore {len(ignore_smiles)} smiles)")

    if env is None:
        # Get unpacking method
        env = BlockMolEnvGraph_v1({"random_steps": 3, "allow_removal": True, "max_steps": 1})

    env_space = env.observation_space["mol_graph"]
    _unpack = env_space.unpack

    next_sample = env_data[list(env_data.keys())[0]]["next"]
    tgt_cl = "dockscore" if "dockscore" in next_sample.columns else "true_dockscore"

    # Construct data
    _inputs, _targets, _smis = [], [], []
    for smi, smi_data in env_data.items():
        if smi in ignore_smiles:
            continue

        candidates = (smi_data["next"].qed_score >= qed_th) & (smi_data["next"].synth_score >= synth_th)

        if dataset_filter_candidates:
            if not candidates.any():
                continue

        x = _unpack(smi_data["obs"]["mol_graph"].astype(np.uint8))
        act_mask = smi_data["obs"]["action_mask"].astype(np.bool)
        act_r_df = smi_data["next"][tgt_cl]
        act_r_df[act_r_df.isna()] = 999999

        if dataset_score_candidates:
            act_r_df[~candidates] = 999999

        # Here we produce targets (based on dockscore)
        tgt = torch.zeros(act_mask.shape)
        tgt[~act_mask] = 999999  # as per logits set
        tgt[act_r_df.index.values] = torch.from_numpy(act_r_df.values).float()
        _smis.append(smi)
        _inputs.append(x)
        _targets.append(tgt)

    print(f"[LOAD DATASET] Total datapoints {len(_inputs)} "
          f"(after filter with candidates {dataset_filter_candidates})")

    # Process Data
    _inputs = [x.to(device) for x in _inputs]
    _targets = torch.stack(_targets)

    # ==============================================================================================
    # Preprocess targets and calculate bins:
    tgt = _targets.numpy()
    bin_values, bins = np.unique(np.round(tgt, binning_decimal), return_inverse=True)
    bins = bins.reshape(tgt.shape)
    bins = torch.from_numpy(bins).to(device)
    print(f"Num unique bins: {len(bin_values)} from {bin_values[0]} to {bin_values[-2]}")

    _targets *= -1  # Invert to have the correct argmax
    # ==============================================================================================

    _targets = _targets.to(device)

    del env_data
    return _inputs, _targets, _smis, bins, env, None


def run(cfg: Namespace):
    set_seed(cfg.seed, cfg.cuda)
    setup_loggers(cfg)
    do_plot = cfg.use_wandb
    out_dir = cfg.out_dir
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    summary_stats = SummaryStats(do_plot)
    eval_loaders = []
    classification = cfg.classification

    # ==============================================================================================
    # Load true dock scores dataset
    test_loader = None
    num_test = 0
    env = None
    ignore_smiles = []
    if getattr(cfg, "dataset_test_filename") is not None:
        t_inputs, t_targets, t_smis, t_bins, env, t_env_data = get_data(
            cfg.dataset_test_filename, torch.device("cpu"), cfg.dataset_cfg
        )
        test_set = CustomDataset(t_inputs, t_targets, t_smis, t_bins, classification=classification, return_extra=True)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, collate_fn=functools.partial(collate_fn, device=device), shuffle=False)
        eval_loaders.append(("test", test_loader))
        if cfg.exclude_test:
            ignore_smiles += t_smis
        num_test = len(t_inputs)

    # ==============================================================================================
    # Load Data

    _inputs, _targets, _smis, bins, env, env_data = get_data(
        cfg.dataset_filename, torch.device("cpu"), cfg.dataset_cfg, env=env, ignore_smiles=ignore_smiles
    )
    cfg.dataset_size = len(_inputs)

    # Create DataLoaders
    rnd_state = np.random.RandomState(cfg.seed_split)
    rand_idxs = rnd_state.permutation(np.arange(len(_inputs)))
    train_idx = rand_idxs[:round(len(_inputs) * cfg.train_split)]
    test_idx = rand_idxs[len(train_idx):]
    cfg.train_size, cfg.test_size = len(train_idx), len(test_idx)

    def get_idxs(idxs):
        return [_inputs[i] for i in idxs], _targets[idxs],  [_smis[i] for i in idxs], bins[idxs]

    train_set = CustomDataset(*get_idxs(train_idx), classification=classification)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, collate_fn=functools.partial(collate_fn, device=device), shuffle=True)

    if len(test_idx) > 0:
        val_set = CustomDataset(*get_idxs(test_idx), classification=classification, return_extra=True)
        val_loader = DataLoader(
            val_set, batch_size=cfg.batch_size, collate_fn=functools.partial(collate_fn, device=device), shuffle=False)
        eval_loaders += [("val", val_loader)]

    print(f"Running on | Train set size: {len(train_idx)} | Eval set size: {len(test_idx)} | Test set size: {num_test} ")

    # ==============================================================================================
    # Load model
    model = get_actor_model(cfg.model)

    model = model.to(device)
    print(model)

    # ==============================================================================================
    # Train

    criterion = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    optimizer = RAdam(model.parameters(), lr=cfg.lr)

    eval_freq = cfg.eval_freq
    log_freq = cfg.log_freq
    training_seen = 0
    best_train_acc = 0
    best_test_acc = 0
    for epoch in range(cfg.num_epochs):  # loop over the dataset multiple times
        acc, loss, other = train_epoch(
            epoch, train_loader, model, optimizer, criterion, device, classification=classification
        )

        if (epoch + 1) % log_freq == 0:
            print(f"[T] E: {epoch} | Acc: {acc*100:.3f}% | los: {loss:.5f}")

        training_seen += len(train_set)

        train_log = {"train_loss": loss, "train_acc": acc*100, "epoch": epoch, "training_seen": training_seen,
                       **other}
        summary_stats.update(train_log)

        wandb_log = copy.deepcopy(train_log)

        if (epoch + 1) % eval_freq == 0 and len(test_idx) > 0:
            for name, _loader in eval_loaders:
                acc, loss, other = eval_epoch(
                    _loader, model, criterion, device, classification=classification
                )
                print(f"\n[E:{name}] E: {epoch} | Acc: {acc * 100:.3f}% | loss: {loss:.5f}\n")

                # Log eval
                eval_log = {
                        "loss": loss, "acc": acc * 100,
                        "epoch": epoch, "training_seen": training_seen,
                        **other
                    }

                for k in list(eval_log.keys()):
                    eval_log[f"{name}_{k}"] = eval_log.pop(k)

                summary_stats.update(eval_log)

                if do_plot:
                    wandb_log.update(eval_log)

        # Log train
        if do_plot:
            wandb.log(wandb_log)

    print("FINISH it"); sys.stdout.flush(); time.sleep(5); os._exit(0)
    sys.exit()
    exit()
    quit()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    from lightrl.utils.utils import parse_opts
    run(parse_opts())

