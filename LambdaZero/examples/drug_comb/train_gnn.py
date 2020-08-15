from LambdaZero.utils import get_external_dirs
from LambdaZero.inputs import random_split
from LambdaZero.examples.drug_comb.model.GNN import GNN
from LambdaZero.examples.drug_comb.new_drugcomb_data_v2 import DrugCombEdge
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from ray import tune
import torch
import ray
import time

def _get_model(config, train_set, val_set):
    model = None
    if config['model'] is GNNWithAttention
        model = GNNWithAttention(config['gcn_channels'], config['rank'],
                                      config['linear_channels'], config['num_relation_lin_lyrs'],
                                      config['gcn_dropout_rate'], config['lin_dropout_rate'],
                                      config['num_residual_gcn_lyrs'], config['aggr'],
                                      train_set.edge_index, val_set.edge_index,
                                      num_relations)
    elif config['model'] is GNN:
        model = GNN(config['gcn_channels'], config['linear_channels'],
                         config['num_relation_lin_lyrs'], config['gcn_dropout_rate'],
                         config['lin_dropout_rate'], config['num_residual_gcn_lyrs'],
                         config['aggr'], train_set.edge_index, val_set.edge_index,
                         num_relations)
    else:
        raise ValueError('Model was not one of GNN or GNNWithAttention')

    return model

def _get_split(dataset, config):
    # If -1 then we use all the examples
    num_examples_to_use = config['num_examples_to_use']

    prop_factor = 1.
    if num_examples_to_use != -1:
        prop_factor = math.max(num_examples_to_use / len(dataset), 1.)

    props = [config['train_prop'], config['val_prop']]
    props = [x * prop_factor for x in props]

    return random_split(len(dataset), props)

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

def run_epoch(loader, model, x_drug, optim, batch_size, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for i, batch in enumerate(loader):
        optim.zero_grad()
        edge_index, edge_classes, y = batch

        y_hat = model(x_drug, edge_index, edge_classes)
        loss = F.mse_loss(y, y_hat)

        loss.backward()
        optim.step()

        metrics['loss'] += loss.item()
        metrics['mse'] += loss.item()
        metrics['mae'] += F.l1_loss(y, y_hat).item()

    for key in metrics.keys():
        metrics[key] /= len(loader)

    return metrics

class DrugDrugGNNRegressor(tune.Trainable):
    def _setup(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = DrugCombEdge(transform=config['transform']).to(device)

        train_idx, val_idx = _get_split(dataset, config)
        train_set = torch.Subset(dataset, train_idx)
        val_set = torch.Subset(dataset, val_idx)

        self.model = _get_model(config, train_set, val_set).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.batch_size = config['batch_size']

        self.x_drugs = dataset.data.x_drugs
        self.train_loader, self.val_loader = _get_loaders(train_set, val_set, config)

    def _train(self):
        train_scores = run_epoch(self.train_loader, self.model, self.x_drugs,
                                 self.optim, self.batch_size, True)

        eval_scores = run_epoch(self.val_loader, self.model, self.x_drugs,
                                self.optim, self.batch_size, False)

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
