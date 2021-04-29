import os
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from torch.nn.functional import mse_loss

from ray import tune
from LambdaZero.utils import MeanVarianceNormalizer


class BasicRegressor(tune.Trainable):
    def setup(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['device'] = self.device
        self.dtype = self.config.get('dtype', torch.float32)
        torch.set_default_dtype(self.dtype)

        # make dataset
        dataset = self.config["dataset"](**self.config["dataset_config"])

        train_idxs, val_idxs, _ = np.load(self.config["dataset_split_path"], allow_pickle=True)
        # fixme!!!!!!!!!!! --- make mpnn index --- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        train_idxs = train_idxs[train_idxs < len(dataset)]
        val_idxs = val_idxs[val_idxs < len(dataset)]
        self.train_set = Subset(dataset, train_idxs.tolist())
        self.val_set = Subset(dataset, val_idxs.tolist())
        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=self.config["batch_size"])
        self.val_loader = DataLoader(self.val_set, batch_size=self.config["batch_size"])

        # patch collate_fn, this is required to allow for dgl
        if hasattr(dataset, 'collate_fn'):
            self.train_loader.collate_fn = dataset.collate_fn
            self.val_loader.collate_fn = dataset.collate_fn

        # make model
        self.model = config["model"](**config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"]["type"](self.model.parameters(), **config["optimizer"]["config"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def step(self):
        train_scores = self.train_epoch(self.config, self.train_loader, self.model, self.optim)
        eval_scores = self.eval_epoch(self.config, self.val_loader, self.model)
        # rename to make scope
        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class RegressorWithSchedulerOnEpoch(BasicRegressor):
    def setup(self, config):
        super().setup(config)
        self.scheduler = config["scheduler"]["type"](self.optim, **config["scheduler"]["config"])
        self.is_scheduler_reduce_lr_on_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def step(self):
        scores = super().step()
        if self.is_scheduler_reduce_lr_on_plateau:
            self.scheduler.step(scores['eval_loss'])  # ReduceLROnPlateau is the only scheduler that accepts metric
        else:
            self.scheduler.step()
        return scores


class RegressorWithSchedulerOnBatch(BasicRegressor):
    def setup(self, config):
        super().setup(config)
        self.scheduler = config["scheduler"]["type"](self.optim, **config["scheduler"]["config"])

    def step(self):
        train_scores = self.train_epoch(self.config, self.train_loader, self.model, self.optim, self.scheduler)
        eval_scores = self.eval_epoch(self.config, self.val_loader, self.model)
        # rename to make scope
        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores


def train_epoch(config, loader, model, optimizer, scheduler=None):
    device = config['device']
    normalizer = MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss": 0}
    epoch_targets = []
    epoch_preds = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        preds_normalized = model(data).view(-1)
        batch_mean_loss = mse_loss(preds_normalized, normalizer.tfm(targets))
        batch_mean_loss.backward()
        optimizer.step()

        # log stuff
        metrics["loss"] += batch_mean_loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        batch_preds = normalizer.itfm(preds_normalized).detach().cpu().numpy()
        epoch_preds.append(batch_preds)
        if scheduler is not None:
            scheduler.step()

    return construct_metrics(metrics, epoch_targets, epoch_preds)


def train_epoch_with_closure(config, loader, model, optimizer, scheduler=None):
    def closure():
        optimizer.zero_grad()
        preds_normalized = model(data).view(-1)
        loss = mse_loss(preds_normalized, normalizer.tfm(targets))
        loss.backward()
        # LBFGS has operation of the form float(closure_output), so usual way of returning multiple outputs fails
        # also, closure is called multiple times for LBFGS, so if any append to the list within it would occur multiple times
        nonlocal batch_preds
        batch_preds = normalizer.itfm(preds_normalized).detach().cpu().numpy()
        return loss

    device = config['device']
    normalizer = MeanVarianceNormalizer(config["target_norm"])
    model.train()

    metrics = {"loss": 0}
    epoch_targets = []
    batch_preds = None
    epoch_preds = []

    for bidx, data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])
        batch_mean_loss = optimizer.step(closure).item()

        # log stuff
        metrics["loss"] += batch_mean_loss * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        epoch_preds.append(batch_preds)
        if scheduler is not None:
            scheduler.step()

    return construct_metrics(metrics, epoch_targets, epoch_preds)


def eval_epoch(config, loader, model):
    device = config['device']
    normalizer = MeanVarianceNormalizer(config["target_norm"])
    model.eval()

    metrics = {"loss": 0}
    epoch_targets = []
    epoch_preds = []

    with torch.no_grad():
        for bidx, data in enumerate(loader):
            data = data.to(device)
            targets = getattr(data, config["target"])
            preds_normalized = model(data).view(-1)
            loss = mse_loss(preds_normalized, normalizer.tfm(targets))

            # log stuff
            metrics["loss"] += loss.item() * data.num_graphs
            epoch_targets.append(targets.detach().cpu().numpy())
            epoch_preds.append(normalizer.itfm(preds_normalized).detach().cpu().numpy())

    return construct_metrics(metrics, epoch_targets, epoch_preds)


def construct_metrics(metrics, epoch_targets, epoch_preds):
    epoch_targets = np.concatenate(epoch_targets, 0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds) ** 2).mean()

    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])
    return metrics
