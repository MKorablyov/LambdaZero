import sys

import torch.nn.functional as F
from tqdm import tqdm


def normalize_metrics_by_dataset_size(metrics: dict, dataset_size: int):
    normalized_metrics = dict()
    for key, value in metrics:
        normalized_metrics[key] = value/dataset_size
    return normalized_metrics


class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, y):
        return (y - self.mean)/self.std

    def denormalize(self, normalized_y):
        return self.std*normalized_y + self.mean


def update_metrics(actual_y, predicted_y, loss, number_of_graphs, metrics):
    metrics["loss"] += loss.item() * number_of_graphs
    metrics["mse"] += ((actual_y - predicted_y) ** 2).sum().item()
    metrics["mae"] += ((actual_y - predicted_y).abs()).sum().item()


def train_epoch(loader, model, optimizer, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    normalizer = Normalizer(mean=target_norm[0], std=target_norm[1])

    model.train()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for data in tqdm(loader, desc="TRAIN", file=sys.stdout):

        data = data.to(device)
        actual_y = getattr(data, target)
        normalized_actual_y = normalizer.normalize(actual_y)
        normalized_predicted_y = model(data)
        predicted_y = normalizer.denormalize(normalized_predicted_y)
        loss = F.mse_loss(normalized_predicted_y, normalized_actual_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_metrics(actual_y, predicted_y, loss, data.num_graphs, metrics)

    metrics = normalize_metrics_by_dataset_size(metrics, len(loader.dataset))
    return metrics


def eval_epoch(loader, model, device, config):
    target = config["targets"][0]
    target_norm = config["target_norms"][0]
    normalizer = Normalizer(mean=target_norm[0], std=target_norm[1])
    model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for data in tqdm(loader, desc="EVAL", file=sys.stdout):

        data = data.to(device)
        actual_y = getattr(data, target)
        normalized_actual_y = normalizer.normalize(actual_y)
        normalized_predicted_y = model(data)
        predicted_y = normalizer.denormalize(normalized_predicted_y)
        loss = F.mse_loss(normalized_predicted_y, normalized_actual_y)

        update_metrics(actual_y, predicted_y, loss, data.num_graphs, metrics)

    metrics = normalize_metrics_by_dataset_size(metrics, len(loader.dataset))
    return metrics
