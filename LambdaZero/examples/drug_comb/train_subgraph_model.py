from LambdaZero.examples.drug_comb.drug_combdb_data import DrugCombDb, to_bipartite_drug_protein_graph
from LambdaZero.examples.drug_comb.subgraph_embedding_model import SubgraphEmbeddingRegressorModel
from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Batch
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
import numpy as np
import torch.nn.functional as F
import ray
import torch
import os
import math
import sys
import time

num_iters = 0

def random_split(num_examples, test_prob, valid_prob):
    nvalid = int(num_examples * valid_prob)
    ntest = int(num_examples * test_prob)
    idx = torch.randperm(num_examples)

    train_idx = idx[ntest + nvalid:]
    val_idx = idx[:nvalid]
    test_idx = idx[:ntest]

    return train_idx, val_idx, test_idx

def train_epoch(ddi_graph, train_idxs, model, optimizer, device, config):
    subgraph_dataset = TensorDataset(ddi_graph.edge_index[:, train_idxs].T,
                                     ddi_graph.edge_classes[train_idxs], ddi_graph.y[train_idxs])

    loader = DataLoader(subgraph_dataset,
                        batch_size=config["batch_size"],
                        pin_memory=device == 'cuda')
    model.train()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for i, drug_drug_batch in enumerate(loader):
        optimizer.zero_grad()

        drug_drug_index, edge_classes, y = drug_drug_batch
        drug_drug_index = drug_drug_index.T

        preds = model(drug_drug_index, ddi_graph.drug_protein_graph, edge_classes)
        loss = F.mse_loss(y, preds)

        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item()
        metrics["mse"] += ((y - preds) ** 2).sum().item()
        metrics["mae"] += ((y - preds).abs()).sum().item()

        global num_iters
        num_iters += 1
        if num_iters % 20 == 0:
            print('Train MSE: %f' % loss.item())

    metrics["loss"] = metrics["loss"] / len(subgraph_dataset)
    metrics["mse"] = metrics["mse"] / len(subgraph_dataset)
    metrics["mae"] = metrics["mae"] / len(subgraph_dataset)
    return metrics

def eval_epoch(dataset, eval_idxs,  model, device, config):
    subgraph_dataset = TensorDataset(ddi_graph.edge_index[:, eval_idxs].T,
                                     ddi_graph.edge_classes[eval_idxs], ddi_graph.y[eval_idxs])

    loader = DataLoader(subgraph_dataset, batch_size=config["batch_size"], pin_memory=True)
    model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for drug_drug_batch in loader:
        drug_drug_index, edge_classes, y = drug_drug_batch
        drug_drug_index = drug_drug_index.T

        batch_drugs = np.unique(drug_drug_index.flatten())
        subgraph_data_list = [ddi_graph.drug_idx_to_graph[drug].to(device) for drug in batch_drugs]
        subgraph_batch = Batch.from_data_list(subgraph_data_list)
        subgraph_batch.batch = subgraph_batch.batch.to(device)

        drug_drug_index = drug_drug_index.to(device)
        edge_classes = edge_classes.to(device)
        y = y.to(device)

        preds = model(drug_drug_index, subgraph_batch, edge_classes)
        loss = F.mse_loss(y, preds)

        metrics["loss"] += loss.item()
        metrics["mse"] += ((y - preds) ** 2).sum().item()
        metrics["mae"] += ((y - preds).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(subgraph_dataset)
    metrics["mse"] = metrics["mse"] / len(subgraph_dataset)
    metrics["mae"] = metrics["mae"] / len(subgraph_dataset)
    print("Eval MSE: %f" % metrics["mse"])
    return metrics

class SubgraphRegressor(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = DrugCombDb(transform=config["transform"], pre_transform=config["pre_transform"])
        self.ddi_graph = dataset[0].to(self.device)

        self.train_idxs, self.val_idxs, self.test_idxs = random_split(self.ddi_graph.edge_index.shape[1],
                                                                      config["test_set_prop"],
                                                                      config["val_set_prop"])

        config["in_channels"] = self.ddi_graph.x.shape[1]
        config["out_channels"] = self.ddi_graph.y.shape[1]
        config["num_cell_lines"] = len(np.unique(self.ddi_graph.edge_classes))

        self.model = SubgraphEmbeddingRegressorModel(config).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.dataset, self.train_idxs, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.dataset, self.val_idxs, self.model, self.device, self.config)

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
    "trainer": SubgraphRegressor,
    "trainer_config": {
        "transform": None,
        "pre_transform": to_bipartite_drug_protein_graph,
        "val_set_prop": 0.1,
        "test_set_prop": 0.1,
        "lr": 0.01,
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "embed_channels": 256,
        "regressor_hidden_channels": 64,
        "num_epochs": 256,
        "batch_size": 64,
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 50,
    "stop": {"training_iteration": 1024},
    "checkpoint_at_end": False,
    "resources_per_trial": {"gpu": 1}
}

if __name__ == "__main__":
    ray.init()

    time_to_sleep = 30
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    analysis = tune.run(
        config["trainer"],
        config=config["trainer_config"],
        stop=config["stop"],
        resources_per_trial=config["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=config["checkpoint_at_end"],
        local_dir=config["summaries_dir"],
        checkpoint_freq=config["checkpoint_freq"]
    )

