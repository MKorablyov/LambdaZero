import os
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from torch.nn.functional import mse_loss

from ray import tune
from LambdaZero.utils import MeanVarianceNormalizer

from abc import ABC, abstractmethod
from collections import OrderedDict


def train(config, dataloader, model, metrics, optimizer, scheduler):
    device = config["device"]
    loss_function = config["loss_function"]
    train_target = config["train_target"]

    model.train()

    scores = {"loss": 0}

    epoch_targets = []
    epoch_preds = []

    for data in dataloader:
        data = data.to(device)
        targets = getattr(data, train_target)

        optimizer.zero_grad()
        preds_normalized = model.core(data)
        targets_normalized = model.scaler.normalize(targets).detach()
        batch_mean_loss = loss_function(preds_normalized, targets_normalized)
        batch_mean_loss.backward()
        optimizer.step()

        scores["loss"] += batch_mean_loss.item() * data.num_graphs
        epoch_targets.append(targets.detach().cpu().numpy())
        batch_preds = model.scaler.denormalize(preds_normalized).detach().cpu().numpy()
        epoch_preds.append(batch_preds)
        if scheduler is not None:
            scheduler.step()

    epoch_targets = np.concatenate(epoch_targets, 0)
    epoch_preds = np.concatenate(epoch_preds, 0)

    # missing
    epoch_targets = {train_target: epoch_targets}

    metric_target_unique_names = set(metric_target for (_, _, metric_target) in metrics)
    missing_metric_target_unique_names = metric_target_unique_names.copy()
    missing_metric_target_unique_names.discard(train_target)

    for missing_target_name in missing_metric_target_unique_names:
        epoch_targets_tmp = []
        for data in dataloader:
            epoch_targets_tmp.append(getattr(data, missing_target_name).detach().cpu().numpy())
        epoch_targets[missing_target_name] = np.concatenate(epoch_targets_tmp, 0)

    for (alias, metric_func, metric_target) in metrics:
        scores[alias] = metric_func(epoch_preds, epoch_targets[metric_target])

    return scores


def eval(config, dataloader, model, metrics):
    device = config['device']

    model.eval()

    scores = {}

    metric_target_unique_names = set(metric_target for (_, _, metric_target) in metrics)
    epoch_targets = {metric_target_name: [] for metric_target_name in metric_target_unique_names}

    epoch_preds = []

    with torch.no_grad():
        for data in dataloader:
            for metric_target_name in metric_target_unique_names:
                epoch_targets[metric_target_name].append(getattr(data, metric_target_name).detach().cpu().numpy())
            data = data.to(device)
            epoch_preds.append(model(data))

    for (alias, metric_func, metric_target) in metrics:
        scores[alias] = metric_func(epoch_preds, epoch_targets[metric_target])

    return scores


class Scaler(ABC, torch.nn.Module):
    @abstractmethod
    def normalize(self, x):
        pass

    @abstractmethod
    def denormalize(self, x_scaled):
        pass

    def forward(self, x_scaled):
        # rerouting forward to denormalize streamlines model execution on inference
        return self.denormalize(x_scaled)


class IdentityScaler(Scaler):
    def __init__(self):
        super().__init__()

    def normalize(self, x):
        return x

    def denormalize(self, x_scaled):
        return x_scaled


class StandardScaler(Scaler):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean',  torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def normalize(self, x):
        x_scaled = (x - self.mean) / self.std
        return x_scaled

    def denormalize(self, x_scaled):
        x = (x_scaled * self.std) + self.mean
        return x


class Trainer(tune.Trainable):
    def setup(self, config):
        self.config = config

        # region resolve dtype and device
        dtype = config.get('dtype', torch.float32)
        device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.config['dtype'] = dtype
        self.config['device'] = device
        torch.set_default_dtype(dtype)
        # endregion

        # region train dataset and metrics
        train_alias = config["train_dataset"]["alias"]
        train_metrics = config["train_dataset"]["metrics"]
        train_batch_size = config["train_dataset"]["batch_size"]
        self.config["loss_function"] = config["train_dataset"]["loss_function"]
        self.config["train_target"] = config["train_dataset"]["target"]

        train_target_scaling_definition = config["train_dataset"].get('scaling', default=None)
        if train_target_scaling_definition is None:
            train_target_scaling = IdentityScaler()
        else:
            train_target_scaling = train_target_scaling_definition["class"](**train_target_scaling_definition["config"])
        self.config["train_target_scaling"] = train_target_scaling

        train_set = config["train_dataset"]["class"](**config["train_dataset"]["config"])
        train_idx_path = config["train_dataset"].get('idx_path', default=None)
        if train_idx_path is not None:
            # Subset does NOT accept numpy array - raises error on type check - hence *.tolist()
            train_idx = np.load(train_idx_path).tolist()
            train_set = Subset(train_set, train_idx)

        train_dataloader = DataLoader(train_set, shuffle=True, batch_size=train_batch_size)
        self.train_entity = (train_alias, train_dataloader, train_metrics)
        # endregion

        # region validation datasets and metrics
        val_entities = []
        for val_entity_definition in config["validation_datasets"]:
            val_alias = val_entity_definition["alias"]
            val_metrics = val_entity_definition["metrics"]
            val_batch_size = val_entity_definition.get('batch_size', default=train_batch_size)
            val_set = val_entity_definition["class"](**val_entity_definition["config"])
            val_idx_path = val_entity_definition.get('idx_path', default=None)
            if val_idx_path is not None:
                val_idx = np.load(val_idx_path).tolist()
                val_set = Subset(val_set, val_idx)
            val_dataloader = DataLoader(val_set, batch_size=val_batch_size)
            val_entities.append((val_alias, val_dataloader, val_metrics))
        self.val_entities = val_entities
        # endregion

        # region model
        model_core = config["model"]["class"](**config["model"]["config"])
        self.model = torch.nn.Sequential(OrderedDict([
            ('core', model_core),
            ('scaler', train_target_scaling)
        ]))
        self.model.to(device)
        # endregion

        # optimizer
        self.optim = config["optimizer"]["class"](self.model.parameters(), **config["optimizer"]["config"])

        # region scheduler
        scheduler_definition = config.get('scheduler', default=None)
        epoch_scheduler = None
        batch_scheduler = None
        scheduler_target = None
        if scheduler_definition is not None:
            scheduler = scheduler_definition["class"](self.optim, **scheduler_definition["config"])
            scheduler_trigger = scheduler_definition.get('trigger', default="epoch")
            if scheduler_trigger == "epoch":
                epoch_scheduler = scheduler
                scheduler_target = scheduler_definition.get('target', default=None)
            elif scheduler_trigger == "batch":
                batch_scheduler = scheduler
            else:
                raise ValueError(f"Scheduler trigger should be either 'epoch' or 'batch'! Encountered {scheduler_trigger}")
        self.epoch_scheduler = epoch_scheduler
        self.batch_scheduler = batch_scheduler
        self.scheduler_target = scheduler_target
        # endregion

        # train / validation epochs
        self.train_epoch = config["train_epoch"]
        self.val_epoch = config["eval_epoch"]

    def step(self):
        train_alias, train_dataloader, train_metrics = self.train_entity
        train_scores = self.train_epoch(self.config, train_dataloader, self.model, train_metrics, self.optim, self.batch_scheduler)
        train_scores = [(f"train_{train_alias}" + k, v) for k, v in train_scores.items()]

        val_scores = []
        for (val_alias, val_dataloader, val_metrics) in self.val_entities:
            val_scores_sub = self.val_epoch(self.config, val_dataloader, self.model, val_metrics)
            val_scores_sub = [(f"val_{val_alias}_" + k, v) for k, v in val_scores_sub.items()]
            val_scores.extend(val_scores_sub)

        scores = dict(train_scores + val_scores)

        # TODO: figure something better than this instance check
        if isinstance(self.epoch_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.epoch_scheduler.step(scores[f"val_{self.scheduler_target}"])
        elif self.epoch_scheduler is not None:
            self.epoch_scheduler.step()

        return scores

    def save_checkpoint(self, checkpoint_dir):
        # TODO: general checkpoint (+optimizer, scheduler, ...)?
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


explicit_config = {
    "trainer": Trainer,
    "trainer_config": {
        "dtype": torch.float64,
        "device": torch.device('cuda:0'),
        "train_dataset": {
            "alias": 'QM9',
            "batch_size": 16,
            "loss_function": None,
            "target": 'y',
            "metrics": None,
            "scaling": {
                "class": StandardScaler,
                "config": {
                    "mean": 2.68,
                    "std": 1.5
                }
            },
            "class": None,
            "config": None,
            "idx_path": None
        },
        "validation_datasets": [
            {
                "alias": None,
                "batch_size": None,
                "metrics": None,
                "idx_path": None,
                "class": None,
                "config": None
            },
{
                "alias": None,
                "batch_size": None,
                "metrics": None,
                "idx_path": None,
                "class": None,
                "config": None
            }
        ],
        "model": {
            "class": None,
            "config": None
        },
        "optimizer": {
            "class": None,
            "config": None
        },
        "scheduler": {
            "class": None,
            "config": None,
            "trigger": 'epoch',
            "target": None
        },
        "train_epoch": train,
        "eval_epoch": eval,
    },

    "summaries_dir": summaries_dir,
    "memory": 8 * 10 ** 9,  # 20 * 10 ** 9,

    "stop": {"training_iteration": 120},
    "resources_per_trial": {
        "cpu": 1,
        "gpu": 1.0
    },
    "keep_checkpoint_num": 2,
    "checkpoint_score_attr": "train_loss",
    "num_samples": 1,
    "checkpoint_at_end": False,
    "checkpoint_freq": 100000,  # 1;  0 is a default in tune.run
}


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
