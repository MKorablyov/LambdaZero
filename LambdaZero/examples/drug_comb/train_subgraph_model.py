from LambdaZero.examples.drug_comb.drug_combdb_data import DrugCombDb, to_drug_induced_subgraphs
from LambdaZero.examples.drug_comb.subgraph_embedding_model import SubgraphEmbeddingRegressorModel
from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Batch
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
import ray
import numpy as np
import torch

def random_split(num_examples, test_prob, valid_prob):
    nvalid = int(num_examples * valid_prob)
    ntest = int(num_examples * test_prob)
    idx = torch.randperm(num_examples)

    train_idx = idx[ntest + nvalid:]
    val_idx = idx[:nvalid]
    test_idx = idx[:ntest]

    return train_idx, val_idx, test_idx

def train_epoch(dataset, train_idxs, model, optimizer, device, config):
    subgraph_dataset = TensorDataset(dataset[0].edge_index[:, train_idxs].T,
                                     dataset[0].edge_classes[train_idxs], dataset[0].y[train_idxs])

    loader = DataLoader(subgraph_dataset, batch_size=config["batch_size"])
    model.train()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for drug_drug_batch in loader:
        optimizer.zero_grad()

        drug_drug_index, edge_classes, y = drug_drug_batch
        drug_drug_index = drug_drug_index.T

        batch_drugs = np.unique(drug_drug_index.flatten())
        subgraph_data_list = [dataset[0].drug_idx_to_graph[drug] for drug in batch_drugs]
        subgraph_batch = Batch.from_data_list(subgraph_data_list)

        preds = model(drug_drug_index, subgraph_batch)
        loss = F.mse_loss(y, preds)

        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item()
        metrics["mse"] += ((y - preds) ** 2).sum().item()
        metrics["mae"] += ((y - preds).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(dataset)
    metrics["mse"] = metrics["mse"] / len(dataset)
    metrics["mae"] = metrics["mae"] / len(dataset)
    return metrics

def eval_epoch(dataset, eval_idxs,  model, device, config):
    subgraph_dataset = TensorDataset(dataset.edge_index[:, eval_idxs],
                                     dataset.edge_classes[eval_idxs], dataset.y[eval_idxs])
    loader = DataLoader(dataset, batch_size=config["batch_size"])
    model.eval()()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for drug_drug_batch in loader:
        drug_drug_index, edge_classes, y = drug_drug_batch

        subgraph_data_list = _get_batch_graph_data_list(dataset, drug_drug_index)
        subgraph_batch = Batch.from_data_list(subgraph_data_list)

        preds = model(drug_drug_index, subgraph_batch)
        loss = F.mse_loss(y, preds)

        metrics["loss"] += loss.item()
        metrics["mse"] += ((y - preds) ** 2).sum().item()
        metrics["mae"] += ((y - preds).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(dataset)
    metrics["mse"] = metrics["mse"] / len(dataset)
    metrics["mae"] = metrics["mae"] / len(dataset)
    return metrics

class SubgraphRegressor(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = DrugCombDb(transform=config["transform"],
                             pre_transform=config["pre_transform"])

        self.train_idxs, self.val_idxs, self.test_idxs = random_split(self.dataset[0].edge_index.shape[1],
                                                                      config["test_set_prop"],
                                                                      config["val_set_prop"])

        config["in_channels"] = self.dataset[0].x.shape[1]
        config["out_channels"] = self.dataset[0].y.shape[1]

        self.model = SubgraphEmbeddingRegressorModel(config)
        self.model.to(self.device)
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
        "pre_transform": to_drug_induced_subgraphs,
        "val_set_prop": 0.1,
        "test_set_prop": 0.1,
        "lr": 0.001,
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "embed_channels": 256,
        "regressor_hidden_channels": 64,
        "num_epochs": 120,
        "batch_size": 64,
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 10000,
    "stop": {"training_iteration": 2},
    "checkpoint_at_end": False,
    "resources_per_trial": {"cpu": 6}
}

if __name__ == "__main__":
    ray.init()

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

