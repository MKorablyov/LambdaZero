from torch_geometric.data import Batch

def _get_batch_graph_data_list(dataset, drug_drug_batch):
    drugs = np.unique(drug_drug_batch.flatten())
    return [dataset.drug_idx_to_graph[drug] for drug in drugs]

def train_epoch(dataset, model, optimizer, device, config):
    loader = DataLoader(dataset, batch_size=config["batch_size"])
    model.train()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for drug_drug_batch in loader:
        optimizer.zero_grad()

        subgraph_data_list = _get_batch_graph_data_list(dataset, drug_drug_batch)
        subgraph_batch = Batch.from_data_list(subgraph_data_list)

        y = drug_drug_batch.y
        preds = model(drug_drug_batch, subgraph_batch)
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

def eval_epoch(dataset, model, device, config):
    loader = DataLoader(dataset, batch_size=config["batch_size"])
    model.eval()()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for drug_drug_batch in loader:
        subgraph_data_list = _get_batch_graph_data_list(dataset, drug_drug_batch)
        subgraph_batch = Batch.from_data_list(subgraph_data_list)

        y = drug_drug_batch.y
        preds = model(drug_drug_batch, subgraph_batch)
        loss = F.mse_loss(y, preds)

        metrics["loss"] += loss.item()
        metrics["mse"] += ((y - preds) ** 2).sum().item()
        metrics["mae"] += ((y - preds).abs()).sum().item()

    metrics["loss"] = metrics["loss"] / len(dataset)
    metrics["mse"] = metrics["mse"] / len(dataset)
    metrics["mae"] = metrics["mae"] / len(dataset)
    return metrics

class BasicRegressor(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = DrugCombDb(transform=config["transform"],
                             pre_transform=config["pre_transform"])

        train_idxs, val_idxs, test_idxs = random_split(len(dataset), config["split_proportions"])
        self.train_set = dataset[torch.tensor(train_idxs)]
        self.val_set = dataset[torch.tensor(val_idxs)]
        self.test_set = dataset[torch.tensor(test_idxs)]

        self.model = SubgraphEmbeddingRegressorModel()
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters, lr=config["lr"])

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.val_set, self.model, self.device, self.config)

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

