from LambdaZero.utils import get_external_dirs
from LambdaZero.inputs import random_split
from LambdaZero.examples.drug_comb.model.gnn import GNN
from LambdaZero.examples.drug_comb.new_drugcomb_data_v2 import DrugCombEdge
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from ray import tune
import numpy as np
import torch
import ray
import time
import os

def _get_model(config, train_set, val_set, num_relations):
    return GNN(config['gcn_channels'], config['rank'],
               config['linear_channels'], config['num_relation_lin_lyrs'],
               config['gcn_dropout_rate'], config['lin_dropout_rate'],
               train_set.edge_index, val_set.edge_index,
               num_relations, config['num_residual_gcn_lyrs'],
               config['gnn_lyr_type'], config['aggr'])

def _get_split(dataset, config):
    # If -1 then we use all the examples
    num_examples_to_use = config['num_examples_to_use']

    prop_factor = 1.
    if num_examples_to_use != -1:
        if num_examples_to_use > len(dataset):
            raise ValueError('num_values_to_use must be at most the size of the dataset')

        prop_factor = min(num_examples_to_use / len(dataset), 1.)

    props = [config['train_prop'], config['val_prop']]
    props = [x * prop_factor for x in props]

    # If we're picking only k samples, add a proportion to the props list
    # so the values add to 1.0.
    if not np.isclose(1.0, sum(props)):
        props.append(1 - props[0] - props[1])

    # First two items are train & val, so only return them
    return random_split(len(dataset), props)[:2]

def _get_loaders(train_set, val_set, batch_size, device):
    train_tensor_set = TensorDataset(train_set.edge_index.T,
                                     train_set.edge_classes,
                                     train_set.css)

    val_tensor_set = TensorDataset(val_set.edge_index.T,
                                   val_set.edge_classes,
                                   val_set.css)

    train_loader = DataLoader(train_tensor_set, batch_size,
                              pin_memory=device == 'cpu', shuffle=True)

    val_loader = DataLoader(val_tensor_set, batch_size,
                            pin_memory=device == 'cpu', shuffle=True)

    return train_loader, val_loader

def run_epoch(loader, model, x_drug, optim, is_train):
    model.train() if is_train else model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for i, batch in enumerate(loader):
        edge_index, edge_classes, y = batch

        y_hat = model(x_drug, edge_index, edge_classes)
        loss = F.mse_loss(y, y_hat)

        metrics['loss'] += loss.item()
        metrics['mse'] += loss.item()
        metrics['mae'] += F.l1_loss(y, y_hat).item()

        if is_train:
            loss.backward()
            optim.step()
            optim.zero_grad()

    for key in metrics.keys():
        metrics[key] /= len(loader)

    return metrics

class DrugDrugGNNRegressor(tune.Trainable):
    def _setup(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = DrugCombEdge().to(device)

        train_idx, val_idx = _get_split(dataset, config)
        train_set = dataset[train_idx]
        val_set = dataset[val_idx]

        self.model = _get_model(config, train_set, val_set, dataset.data.num_relations).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])

        self.x_drugs = dataset.data.x_drugs
        self.train_loader, self.val_loader = _get_loaders(train_set, val_set,
                                                          config['batch_size'], device)

    def _train(self):
        train_scores = run_epoch(self.train_loader, self.model, self.x_drugs,
                                 self.optim, True)

        eval_scores = run_epoch(self.val_loader, self.model, self.x_drugs,
                                self.optim, False)

        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]

        scores = dict(train_scores + eval_scores)
        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

_, _, summaries_dir =  get_external_dirs()
config = {
    "trainer": DrugDrugGNNRegressor,
    "trainer_config": {
        "model": GNN,
        "gcn_channels": [1024, 512, 512],
        "rank": 124,
        "linear_channels": [1024, 512, 256, 128, 1],
        "num_relation_lin_lyrs": 2,
        "gcn_dropout_rate": .1,
        "lin_dropout_rate": .3,
        "num_residual_gcn_lyrs": 1,
        "aggr": "concat",
        "num_examples_to_use": 100,
        "train_prop": .8,
        "val_prop": .2,
        "lr": 1e-4,
        "batch_size": 32,
        "gnn_lyr_type": "GCNWithAttention", # Must be a str as we can't pickle modules
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 200,
    "stop": {"training_iteration": 100000},
    "checkpoint_at_end": False,
    "resources_per_trial": {"gpu": 1},
    "name": "Testing"
}

if __name__ == "__main__":
    ray.init()

    time_to_sleep = 5
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    analysis = tune.run(
        config["trainer"],
        name=config["name"],
        config=config["trainer_config"],
        stop=config["stop"],
        resources_per_trial=config["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=config["checkpoint_at_end"],
        local_dir=config["summaries_dir"],
        checkpoint_freq=config["checkpoint_freq"]
    )
