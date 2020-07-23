from LambdaZero.examples.drug_comb.drug_combdb_data import DrugCombDb, to_drug_induced_subgraphs, subgraph_protein_features_to_embedding, use_score_type_as_target, use_single_cell_line
from LambdaZero.examples.drug_comb.subgraph_embedding_model import SubgraphEmbeddingRegressorModel
from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Batch
from torchvision.transforms import Compose
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
                        pin_memory=device == 'cpu',
                        shuffle=True)
    model.train()

    num_batches = math.ceil(ddi_graph.edge_index.shape[1] / config["batch_size"])
    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for i, drug_drug_batch in enumerate(loader):
        optimizer.zero_grad()

        drug_drug_index, edge_classes, y = drug_drug_batch
        drug_drug_index = drug_drug_index.T

        # Here, sg is shorthand for subgraph
        sg_edge_index, sg_nodes, sg_avging_idx = get_batch_subgraph_edges(ddi_graph, drug_drug_index)

        preds = model(ddi_graph.protein_ftrs, drug_drug_index, edge_classes,
                      sg_edge_index, sg_nodes, sg_avging_idx)
        loss = F.mse_loss(y, preds)

        loss.backward()
        optimizer.step()

        if math.isnan(loss.item()):
            for k, v in metrics.items():
                metrics[k] = (v * num_batches) / (num_batches - 1)

            num_batches -= 1

        else:
            metrics["loss"] += loss.item() / num_batches
            metrics["mse"] += loss.item() / num_batches
            metrics["mae"] += F.l1_loss(y, preds).item() / num_batches

        global num_iters
        num_iters += 1
        if num_iters % 40 == 0:
            print('Train MSE: %f' % loss.item())

    return metrics

def eval_epoch(ddi_graph, eval_idxs,  model, device, config):
    subgraph_dataset = TensorDataset(ddi_graph.edge_index[:, eval_idxs].T,
                                     ddi_graph.edge_classes[eval_idxs], ddi_graph.y[eval_idxs])

    loader = DataLoader(subgraph_dataset, batch_size=config["batch_size"], pin_memory=device == 'cpu')
    model.eval()

    num_batches = math.ceil(ddi_graph.edge_index.shape[1] / config["batch_size"])
    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for drug_drug_batch in loader:
        drug_drug_index, edge_classes, y = drug_drug_batch
        drug_drug_index = drug_drug_index.T

        # Here, sg is shorthand for subgraph
        sg_edge_index, sg_nodes, sg_avging_idx = get_batch_subgraph_edges(ddi_graph, drug_drug_index)

        preds = model(ddi_graph.protein_ftrs, drug_drug_index, edge_classes,
                      sg_edge_index, sg_nodes, sg_avging_idx)
        loss = F.mse_loss(y, preds)

        if math.isnan(loss.item()):
            for k, v in metrics.items():
                metrics[k] = (v * num_batches) / (num_batches - 1)

            num_batches -= 1

        else:
            metrics["loss"] += loss.item() / num_batches
            metrics["mse"] += loss.item() / num_batches
            metrics["mae"] += F.l1_loss(y, preds).item() / num_batches

    print("Eval MSE: %f" % metrics["mse"])
    return metrics

def get_batch_subgraph_edges(ddi_graph, batch_drug_drug_index):
    batch_drugs = torch.unique(batch_drug_drug_index.flatten())
    subgraph_edge_indices = tuple([
        ddi_graph.drug_idx_to_graph[drug.item()].edge_index for drug in batch_drugs
    ])

    edge_index = torch.cat(subgraph_edge_indices, dim=1)
    edge_index = torch.unique(edge_index, dim=1, sorted=False)

    subgraph_nodes = tuple([ddi_graph.drug_idx_to_graph[drug.item()].nodes for drug in batch_drugs])
    nodes = torch.cat(subgraph_nodes)

    averaging_idx_tensors = tuple([
        torch.full((len(nodes),), i, dtype=torch.long) for i, nodes in enumerate(subgraph_nodes)
    ])

    averaging_idx = torch.cat(averaging_idx_tensors).to(batch_drugs.device)

    return edge_index, nodes, averaging_idx

class SubgraphRegressor(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform_list = [use_score_type_as_target(config['score_type'])]
        if not config['use_one_hot']:
            transform_list.append(
                subgraph_protein_features_to_embedding(config['protein_embedding_size'], self.device)
            )

        if config['use_single_cell_line']:
            transform_list.append(use_single_cell_line)

        dataset = DrugCombDb(transform=Compose(transform_list), pre_transform=config["pre_transform"])
        self.ddi_graph = self._extract_ddi_graph(dataset)

        self.train_idxs, self.val_idxs, self.test_idxs = random_split(self.ddi_graph.edge_index.shape[1],
                                                                      config["test_set_prop"],
                                                                      config["val_set_prop"])

        config["in_channels"] = self.ddi_graph.x.shape[1] if config['use_one_hot'] else config["protein_embedding_size"]
        config["out_channels"] = self.ddi_graph.y.shape[1] if len(self.ddi_graph.y.shape) > 1 else 1

        self.model = SubgraphEmbeddingRegressorModel(
            config, torch.unique(self.ddi_graph.edge_classes)).to(self.device)

        to_optimize = list(self.model.parameters())
        if config['use_one_hot'] == False:
            to_optimize.append(self.ddi_graph.protein_ftrs)

        self.optim = torch.optim.Adam(to_optimize, lr=config["lr"])

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _extract_ddi_graph(self, dataset):
        ddi_graph = dataset[0].to(self.device)
        ddi_graph.protein_ftrs = ddi_graph.protein_ftrs.to(self.device)
        for drug, subgraph in ddi_graph.drug_idx_to_graph.items():
            new_subgraph = subgraph.to(self.device)
            if hasattr(new_subgraph, 'nodes'):
                new_subgraph.nodes = subgraph.nodes.to(self.device)

            ddi_graph.drug_idx_to_graph[drug] = new_subgraph

        return ddi_graph

    def _train(self):
        train_scores = self.train_epoch(self.ddi_graph, self.train_idxs, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.ddi_graph, self.val_idxs, self.model, self.device, self.config)

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
        "pre_transform": to_drug_induced_subgraphs,
        "val_set_prop": 0.2,
        "test_set_prop": 0.0,
        "prediction_type": tune.grid_search(["dot_product", "mlp"]),
        "lr": tune.grid_search([1e-4, 5e-4, 5e-5, 1e-5]),
        "use_one_hot": tune.grid_search([True, False]),
        "score_type": 'hsa',
        "weight_initialization_type": "torch_base",
        "protein_embedding_size": 256,
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "embed_channels": 256,
        "regressor_hidden_channels": 64,
        "num_epochs": 256,
        "batch_size": 64,
        "conv_dropout_rate": 0.1,
        "linear_dropout_rate": 0.2,
        "use_single_cell_line": tune.grid_search([True, False]),
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 200,
    "stop": {"training_iteration": 250},
    "checkpoint_at_end": True,
    "resources_per_trial": {"gpu": 1},
    "name": None
}

if __name__ == "__main__":
    ray.init()

    time_to_sleep = 5
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    config["name"] = "ShareFeaturesSingleCellLine"
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







