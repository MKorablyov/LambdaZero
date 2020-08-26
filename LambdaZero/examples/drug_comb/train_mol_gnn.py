from LambdaZero.utils import get_external_dirs, MeanVarianceNormalizer
from LambdaZero.inputs import random_split
from LambdaZero.examples.drug_comb.model.mol_gnn import MolGnnPredictor
from LambdaZero.examples.drug_comb.new_drugcomb_data_v2 import DrugCombEdge
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import numpy as np
from hyperopt import hp
import torch
import ray
import time
import os
import math

def _get_model(config, train_set, num_relations):
    return MolGnnPredictor(config['linear_channels'], config['num_relation_lin_lyrs'],
                           int(config['embed_dim']), config['gcn_dropout_rate'],
                           config['lin_dropout_rate'], num_relations,
                           train_set.mol_graphs[0].x.shape[1], config['aggr'])

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
    return dataset.random_split(props[1], .0)[:2]

def _get_loaders(train_set, val_set, batch_size, device):
    train_tensor_set = TensorDataset(train_set.edge_index.T,
                                     train_set.edge_classes,
                                     torch.cat((train_set.row_ic50, train_set.col_ic50), dim=1),
                                     train_set.css)

    val_tensor_set = TensorDataset(val_set.edge_index.T,
                                   val_set.edge_classes,
                                   torch.cat((val_set.row_ic50, val_set.col_ic50), dim=1),
                                   val_set.css)

    train_loader = DataLoader(train_tensor_set, batch_size,
                              pin_memory=device == 'cpu', shuffle=True)

    val_loader = DataLoader(val_tensor_set, batch_size,
                            pin_memory=device == 'cpu', shuffle=True)

    return train_loader, val_loader

def run_epoch(loader, model, normalizer, graphs, optim, is_train):
    model.train() if is_train else model.eval()

    loss_sum      = 0
    epoch_targets = []
    epoch_preds   = []
    for i, batch in enumerate(loader):
        edge_index, edge_classes, edge_attr, y = batch
        edge_index, graph_batch = _rebuild_edge_index_and_graphs(edge_index, graphs)

        y_hat = model(graph_batch, edge_index, edge_classes, edge_attr)
        loss = F.mse_loss(normalizer.tfm(y), y_hat)

        if is_train:
            loss.backward()
            optim.step()
            optim.zero_grad()

        loss_sum += loss.item()
        epoch_targets.append(y.detach().cpu().numpy())
        epoch_preds.append(normalizer.itfm(y_hat).detach().cpu().numpy())

    epoch_targets = torch.tensor(np.concatenate(epoch_targets, axis=0))
    epoch_preds = torch.tensor(np.concatenate(epoch_preds, axis=0))

    return {
        "loss": loss_sum / epoch_targets.shape[0],
        "mae": F.l1_loss(epoch_targets, epoch_preds).item(),
        "mse": F.mse_loss(epoch_targets, epoch_preds).item(),
        "rmse": torch.sqrt(F.mse_loss(epoch_targets, epoch_preds)).item(),
    }

def _rebuild_edge_index_and_graphs(edge_index, graphs):
    '''
    Since running all graphs through MPNN would drastically OOM (have observed
    just sending 32 graphs in OOMing, and there's 4200 total in the dataset!),
    we need to get the specific graphs for the batch and reindex the edge_index
    to match that of the new list of graphs.  This method does both things.
    '''
    # torch.unique guarantees sorted return vals
    # Load batch into memory one at a time to save on memory consumption
    drug_idxs = torch.unique(edge_index, sorted=True)
    graph_batch = Batch.from_data_list([graphs[i] for i in drug_idxs]).to(edge_index.device)

    # Re-index edge_index relative to graph idxs
    bins = np.unique(edge_index.cpu().flatten()) + 1
    re_edge_index = torch.from_numpy(np.digitize(edge_index.cpu(), bins)).to(edge_index.device)

    return re_edge_index, graph_batch

class DrugDrugMolGNNRegressor(tune.Trainable):
    def _setup(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = DrugCombEdge().to(device)

        train_idx, val_idx = _get_split(dataset, config)
        train_set = dataset[train_idx]
        val_set = dataset[val_idx]

        self.model = _get_model(config, train_set, dataset.data.num_relations).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.graphs = dataset.data.mol_graphs

        self.train_loader, self.val_loader = _get_loaders(train_set, val_set,
                                                          config['batch_size'], device)

        # Base variance for computing explained variance
        self.var0 = F.mse_loss(val_set.css, train_set.css.mean()).item()
        self.normalizer = MeanVarianceNormalizer((dataset[:].css.mean(), dataset[:].css.var()))

    def _train(self):
        train_scores = run_epoch(self.train_loader, self.model,
                                 self.normalizer, self.graphs,
                                 self.optim, True)

        eval_scores = run_epoch(self.val_loader, self.model,
                                self.normalizer, self.graphs,
                                self.optim, False)

        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]

        scores = dict(train_scores + eval_scores)

        # Add explained variance
        scores['explained_variance'] = (self.var0 - scores['eval_mse']) / self.var0

        for k, v in scores.items():
            if math.isnan(v):
                scores[k] = float('inf')

        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

_, _, summaries_dir =  get_external_dirs()
config = {
    "trainer": DrugDrugMolGNNRegressor,
    "trainer_config": {
        "model": MolGnnPredictor,
        "linear_channels": [1024, 512, 256, 128, 1],
        "num_relation_lin_lyrs": 2,
        "embed_dim": 64,
        "num_examples_to_use": -1,
        "train_prop": .8,
        "val_prop": .2,
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 200,
    "stop": {"training_iteration": 100},
    "checkpoint_at_end": False,
    "resources_per_trial": {"gpu": 1},
    "name": "MPNNMolStructure",
    "asha_metric": "eval_mse",
    "asha_mode": "min",
    "asha_max_t": 100
}

if __name__ == "__main__":
    ray.init()

    time_to_sleep = 15
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    asha_scheduler = ASHAScheduler(
	time_attr='training_iteration',
	metric=config['asha_metric'],
	mode=config['asha_mode'],
	max_t=config['asha_max_t'],
	grace_period=10,
	reduction_factor=3,
	brackets=1
    )

    search_space = {
        "lr": hp.loguniform("lr", np.log(1e-7), np.log(5e-3)),
        "batch_size": hp.choice("batch_size", [256, 512, 1028]),
        "gcn_dropout_rate": hp.uniform("gcn_dropout_rate", .0, .2),
        "lin_dropout_rate": hp.uniform("lin_dropout_rate", .0, .4),
        "aggr": hp.choice("aggr", ["concat", "hadamard"]),
    }

    current_best_params = [
        {
            "lr": 1e-4,
            "batch_size": 1,
            "gcn_dropout_rate": .1,
            "lin_dropout_rate": .4,
            "aggr": 0,
        }
    ]

    search_alg = HyperOptSearch(
        search_space,
        metric=config['asha_metric'],
        mode=config['asha_mode'],
        points_to_evaluate=current_best_params
    )

    analysis = tune.run(
        config["trainer"],
        name=config["name"],
        config=config["trainer_config"],
        stop=config["stop"],
        resources_per_trial=config["resources_per_trial"],
        num_samples=1000,
        checkpoint_at_end=config["checkpoint_at_end"],
        local_dir=config["summaries_dir"],
        checkpoint_freq=config["checkpoint_freq"],
        scheduler=asha_scheduler,
        search_alg=search_alg,
    )
