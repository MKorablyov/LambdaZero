from LambdaZero.utils import get_external_dirs, MeanVarianceNormalizer
from LambdaZero.inputs import random_split
from LambdaZero.examples.drug_comb.model.mol_gnn import MolGnnPredictor
from LambdaZero.examples.drug_comb.new_drugcomb_data_v2 import DrugCombEdge
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
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

def _get_model(config, train_set, val_set, num_relations):
    return MolGnnPredictor(config['linear_channels'], config['num_relation_lin_lyrs'],
                           int(config['embed_dim']), config['gcn_dropout_rate'],
                           config['lin_dropout_rate'], num_relations,
                           train_set.mol_graphs, val_set.mol_graphs)

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

def run_epoch(loader, model, normalizer, optim, is_train):
    model.train() if is_train else model.eval()

    loss_sum      = 0
    epoch_targets = []
    epoch_preds   = []
    for i, batch in enumerate(loader):
        edge_index, edge_classes, edge_attr, y = batch

        y_hat = model(edge_index, edge_classes, edge_attr)
        loss = F.mse_loss(normalizer.tfm(y), y_hat)

        if is_train:
            loss.backward()
            optim.step()
            optim.zero_grad()

        loss_sum += loss.item()
        epoch_targets.append(y)
        epoch_preds.append(normalizer.itfm(y_hat))

    epoch_targets = torch.cat(epoch_targets)
    epoch_preds = torch.cat(epoch_preds)

    return {
        "loss": loss_sum / epoch_targets.shape[0],
        "mae": F.l1_loss(epoch_targets, epoch_preds).item(),
        "mse": F.mse_loss(epoch_targets, epoch_preds).item(),
        "rmse": torch.sqrt(F.mse_loss(epoch_targets, epoch_preds)).item(),
    }

class DrugDrugMolGNNRegressor(tune.Trainable):
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

        # Base variance for computing explained variance
        self.var0 = F.mse_loss(val_set.css, train_set.css.mean()).item()
        self.normalizer = MeanVarianceNormalizer((dataset[:].css.mean(), dataset[:].css.var()))

    def _train(self):
        train_scores = run_epoch(self.train_loader, self.model, self.normalizer,
                                 self.x_drugs, self.optim, True)

        eval_scores = run_epoch(self.val_loader, self.model, self.normalizer,
                                self.x_drugs, self.optim, False)

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
    "trainer": DrugDrugGNNRegressor,
    "trainer_config": {
        "model": MolGnnPredictor,
        "linear_channels": [1024, 512, 256, 128, 1],
        "num_relation_lin_lyrs": 2,
        "num_residual_gcn_lyrs": 1,
        "aggr": "concat",
        "num_examples_to_use": 10,
        "train_prop": .8,
        "val_prop": .2,
        "gnn_lyr_type": "GCNWithAttention",
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 200,
    "stop": {"training_iteration": 100},
    "checkpoint_at_end": False,
    "resources_per_trial": {"gpu": 1},
    "name": "DrugCombTryHyperopt",
    "asha_metric": "eval_mse",
    "asha_mode": "min",
    "asha_max_t": 100
}

if __name__ == "__main__":
    ray.init()

    time_to_sleep = 5
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
        "lr": hp.loguniform("lr", np.log(1e-7), np.log(5e-2)),
        "batch_size": hp.choice("batch_size", [256, 512]),
        "embed_dim": hp.quniform("embed_dim", 32, 512, 1)
        "gcn_dropout_rate": hp.uniform("gcn_dropout_rate", .0, .2),
        "lin_dropout_rate": hp.uniform("lin_dropout_rate", .0, .4),
    }

    current_best_params = [
        {
            "lr": 1e-4,
            "batch_size": 0,
            "embed_dim": 64,
            "gcn_dropout_rate": .1,
            "lin_dropout_rate": .4,
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
        num_samples=100000,
        checkpoint_at_end=config["checkpoint_at_end"],
        local_dir=config["summaries_dir"],
        checkpoint_freq=config["checkpoint_freq"],
        scheduler=asha_scheduler,
        search_alg=search_alg,
    )
