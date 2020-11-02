import time
import ray
import gc
import torch
import sklearn.metrics
import numpy as np
import os
from hyperopt import hp
from torch_geometric.transforms import Compose
from torch_geometric.nn import GCNConv
from LambdaZero.utils import get_external_dirs
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from LambdaZero.examples.drug_comb.utils import IntParamRegistrar
from LambdaZero.examples.drug_comb.datasets.utils import specific_cell_line, take_first_k_vals
from LambdaZero.examples.drug_comb.datasets.drugcomb_data_score import DrugCombScore
from LambdaZero.examples.drug_comb.datasets.drugcomb_score_l1000_data import DrugCombScoreL1000NoPPI
from LambdaZero.examples.drug_comb.models.models import GiantGraphGCN, GraphSignalLearner
from LambdaZero.examples.drug_comb.models.message_conv_layers import FourMessageConvLayer
from LambdaZero.examples.drug_comb.models.predictors import SharedLayersMLPPredictor, FilmMLPPredictor, PPIPredictor
from LambdaZero.examples.drug_comb.models.acquisition import RandomAcquisition, ExpectedImprovement, GreedyAcquisition
from LambdaZero.examples.drug_comb.models.pooling import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    GlobalAttention,
    Set2Set,
    SAGPooling,
    TopKPooling,
)

########################################################################################################################
# Epoch loops
########################################################################################################################


def train_epoch(data, loader, model, optim, max_examp_per_epoch=None, n_forward_passes=1):
    model.train()
    model.enable_periodic_backprop()

    epoch_loss = 0
    num_batches = len(loader)

    examples_seen = 0

    for i, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch, n_forward_passes=n_forward_passes)
        loss = model.loss(out, drug_drug_batch)

        loss.backward()
        optim.step()

        epoch_loss += loss.item()

        # If we have seen enough examples in this epoch, break
        examples_seen += drug_drug_batch[0].shape[0]
        if max_examp_per_epoch is not None:
            if examples_seen >= max_examp_per_epoch:
                break

    print('Mean train loss: {:.4f}'.format(epoch_loss / num_batches))

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches}


def eval_epoch(data, loader, model, acquisition=None, n_forward_passes=1):
    # const
    TARGET_IDX = 3

    model.eval()
    model.disable_periodic_backprop()
    if acquisition is not None:
        model.enable_dropout()

    epoch_loss = 0
    r_squared = 0
    num_batches = len(loader)
    active_scores = torch.empty(0)
    epoch_targets = []
    epoch_preds = []

    with torch.no_grad():
        for i, drug_drug_batch in enumerate(loader):
            out = model.forward(data, drug_drug_batch, n_forward_passes=n_forward_passes)

            if acquisition is not None:
                active_scores = torch.cat((active_scores, acquisition.get_scores(out)))

            loss = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()

            # Explained variance
            epoch_targets.append(drug_drug_batch[TARGET_IDX].detach().cpu().numpy())
            epoch_preds.append(out.detach().cpu().numpy())

    print('Mean valid loss: {:.4f}'.format(epoch_loss / num_batches))

    epoch_targets = np.concatenate(epoch_targets, 0)
    epoch_preds = np.concatenate(epoch_preds, 0)

    summary_dict = {
        "loss_sum": epoch_loss,
        "loss_mean": epoch_loss / num_batches,
        "r_squared": sklearn.metrics.r2_score(epoch_targets, epoch_preds)
    }

    if acquisition is not None:
        return summary_dict, active_scores

    return summary_dict

########################################################################################################################
# Abstract trainer
########################################################################################################################


class AbstractTrainer(tune.Trainable):
    def _setup(self, config):
        self.batch_size = config["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_train_forward_passes = config["n_train_forward_passes"]
        config["device"] = self.device

        dataset = config["dataset"](
            transform=config["transform"],
            pre_transform=config["pre_transform"],
            fp_bits=config["fp_bits"],
            fp_radius=config["fp_radius"],
            ppi_confidence_thres=config["ppi_confidence_thres"],
        )

        self.data = dataset[0].to(self.device)

        # If a score dataset is used, we have to specify the target
        if "target" in config.keys():
            possible_target_dicts = {
                "css": self.data.ddi_edge_css,
                "bliss": self.data.ddi_edge_bliss,
                "zip": self.data.ddi_edge_zip,
                "hsa": self.data.ddi_edge_hsa,
                "loewe": self.data.ddi_edge_loewe,
            }
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])

        self.train_idxs, self.val_idxs, self.test_idxs = dataset.random_split(
            config["test_set_prop"], config["val_set_prop"], self.data.ddi_edge_idx
        )

        # Valid loader
        valid_ddi_dataset = TensorDataset(
            self.data.ddi_edge_idx[:, self.val_idxs].T,
            self.data.ddi_edge_classes[self.val_idxs],
            self.data.ddi_edge_attr[self.val_idxs],
            self.data.ddi_edge_response[self.val_idxs],
        )

        self.valid_loader = DataLoader(
            valid_ddi_dataset,
            batch_size=config["batch_size"],
            pin_memory=(self.device == "cpu"),
        )

        # Initialize model and optimizer
        self.model = config["model"](self.data, config).to(self.device)
        print(self.model)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        raise NotImplementedError

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


########################################################################################################################
# Basic trainer
########################################################################################################################


class BasicTrainer(AbstractTrainer):
    def _setup(self, config):
        print("Initializing regular training pipeline")
        super(BasicTrainer, self)._setup(config)
        # Train loader
        train_ddi_dataset = TensorDataset(
            self.data.ddi_edge_idx[:, self.train_idxs].T,
            self.data.ddi_edge_classes[self.train_idxs],
            self.data.ddi_edge_attr[self.train_idxs],
            self.data.ddi_edge_response[self.train_idxs],
        )

        self.train_loader = DataLoader(
            train_ddi_dataset,
            batch_size=config["batch_size"],
            pin_memory=(self.device == "cpu"),
        )

    def _train(self):
        train_metrics = self.train_epoch(
            self.data,
            self.train_loader,
            self.model,
            self.optim,
            n_forward_passes=self.n_train_forward_passes,
        )
        eval_metrics = self.eval_epoch(self.data, self.valid_loader, self.model)

        train_metrics = [("train_" + k, v) for k, v in train_metrics.items()]
        eval_metrics = [("eval_" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(train_metrics + eval_metrics)

        return metrics


########################################################################################################################
# Active Trainer
########################################################################################################################


class ActiveTrainer(AbstractTrainer):
    def _setup(self, config):
        print("Initializing active training pipeline")
        super(ActiveTrainer, self)._setup(config)

        self.acquire_n_at_a_time = config["acquire_n_at_a_time"]
        self.max_examp_per_epoch = config["max_examp_per_epoch"]
        self.acquisition = config["acquisition"](config)
        self.n_scoring_forward_passes = config["n_scoring_forward_passes"]
        self.n_epoch_between_queries = config["n_epoch_between_queries"]

        # randomly acquire data at the beginning
        self.seen_idxs = self.train_idxs[:self.acquire_n_at_a_time]
        self.unseen_idxs = self.train_idxs[self.acquire_n_at_a_time:]
        # Initialize variable that saves the last query
        self.last_query_idxs = self.seen_idxs

        self.seen_loader, self.unseen_loader, self.last_query_loader = self.update_loaders(self.seen_idxs,
                                                                                           self.unseen_idxs,
                                                                                           self.last_query_idxs)
        # For top 1 percent
        one_perc = int(0.01 * len(self.unseen_idxs))
        scores = self.data.ddi_edge_response[self.unseen_idxs]
        self.top_one_perc = set(self.unseen_idxs[torch.argsort(scores, descending=True)[:one_perc]].numpy())
        self.count = 0

    def _train(self):
        # Evaluate on last query before training
        last_query_before_metric = self.eval_epoch(self.data, self.last_query_loader, self.model)

        # Train on seen examples
        for _ in range(self.n_epoch_between_queries):
            # Perform several training epochs. Save only metrics from the last epoch
            seen_metrics = self.train_epoch(self.data, self.seen_loader, self.model, self.optim,
                                            self.max_examp_per_epoch, n_forward_passes=self.n_train_forward_passes)

        # Evaluate on last query after training
        last_query_after_metric = self.eval_epoch(self.data, self.last_query_loader, self.model)

        # Evaluate
        unseen_metrics, active_scores = self.eval_epoch(self.data, self.unseen_loader, self.model, self.acquisition,
                                                        self.n_scoring_forward_passes)
        eval_metrics = self.eval_epoch(self.data, self.valid_loader, self.model)

        seen_metrics = [("seen_" + k, v) for k, v in seen_metrics.items()]
        last_query_before_metric = [("last_query_before_tr_" + k, v) for k, v in last_query_before_metric.items()]
        last_query_after_metric = [("last_query_after_tr_" + k, v) for k, v in last_query_after_metric.items()]
        unseen_metrics = [("unseen_" + k, v) for k, v in unseen_metrics.items()]
        eval_metrics = [("eval_" + k, v) for k, v in eval_metrics.items()]
        metrics = dict(seen_metrics + unseen_metrics + eval_metrics +
                       last_query_before_metric + last_query_after_metric)

        # Acquire new data
        query = self.unseen_idxs[torch.argsort(active_scores, descending=True)[:self.acquire_n_at_a_time]]
        # remove the query from the unseen examples
        self.unseen_idxs = self.unseen_idxs[torch.argsort(active_scores, descending=True)[self.acquire_n_at_a_time:]]
        # Add the query to the seen examples
        self.seen_idxs = torch.cat((self.seen_idxs, query))

        # For top 1 percent
        query_set = set(query.detach().numpy())
        self.count += len(query_set & self.top_one_perc)
        metrics["top"] = self.count / len(self.top_one_perc)

        self.last_query_idxs = query

        self.seen_loader, self.unseen_loader, self.last_query_loader = self.update_loaders(self.seen_idxs,
                                                                                           self.unseen_idxs,
                                                                                           self.last_query_idxs)
        return metrics

    def update_loaders(self, seen_idxs, unseen_idxs, last_query_idxs):
        # Seen loader
        seen_ddi_dataset = TensorDataset(self.data.ddi_edge_idx[:, seen_idxs].T,
                                         self.data.ddi_edge_classes[seen_idxs], self.data.ddi_edge_attr[seen_idxs],
                                         self.data.ddi_edge_response[seen_idxs])

        seen_loader = DataLoader(seen_ddi_dataset,
                                 batch_size=self.batch_size,
                                 pin_memory=(self.device == 'cpu'))

        # Unseen loader
        unseen_ddi_dataset = TensorDataset(self.data.ddi_edge_idx[:, unseen_idxs].T,
                                           self.data.ddi_edge_classes[unseen_idxs],
                                           self.data.ddi_edge_attr[unseen_idxs],
                                           self.data.ddi_edge_response[unseen_idxs])

        unseen_loader = DataLoader(unseen_ddi_dataset,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   pin_memory=(self.device == 'cpu'))

        # Last query loader
        last_query_ddi_dataset = TensorDataset(self.data.ddi_edge_idx[:, last_query_idxs].T,
                                               self.data.ddi_edge_classes[last_query_idxs],
                                               self.data.ddi_edge_attr[last_query_idxs],
                                               self.data.ddi_edge_response[last_query_idxs])

        last_query_loader = DataLoader(last_query_ddi_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=(self.device == 'cpu'))

        # Update acquisition function
        self.acquisition.update_with_seen(self.data.ddi_edge_response[seen_idxs])

        return seen_loader, unseen_loader, last_query_loader


if __name__ == "__main__":

    pipeline_config = {
        "transform": Compose([specific_cell_line('K-562'), take_first_k_vals(10)]),
        "pre_transform": None,
        "seed": 1,  # tune.grid_search([1, 2, 3]),
        "val_set_prop": 0.2,
        "test_set_prop": 0.0,
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "batch_size": 1,  # tune.grid_search([1024, 2048]),
        "weight_decay": 0,
    }

    predictor_config = {
        "predictor": PPIPredictor,
        "predictor_layers": [1024, 512, 256, 1],
        # [1024, 512, 256, 1],  # tune.grid_search([[2048, 1024, 1], [4096, 2048, 1024, 1]]),
        "with_fp": False,
        "with_expr": True,
        "with_prot": True,
        "average_prot_embeddings": True,
    }

    model_config = {
        # model: GiantGraphGCN, Baseline or Dummy
        "model": GraphSignalLearner,  # tune.grid_search([MLPBaselineFPExpr, MLPBaselineFPProt]),
        "attention": True,  # tune.grid_search([False, True]),
        "pass_d2d_msg": True,
        "pass_d2p_msg": False,
        "pass_p2d_msg": False,
        "pass_p2p_msg": False,
        "do_periodic_backprop": False,
        'residual_block_length': 1,
        'pooling_gnn': GCNConv,
        'attention_all_layers': True,
        'num_set2set_steps': 5,
    }

    dataset_config = {
        "dataset": DrugCombScore,
        "target": "zip",  # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
        "fp_bits": 1024,
        "fp_radius": 4,
        "ppi_confidence_thres": 0.9,
    }

    active_learning_config = {
        "acquisition": ExpectedImprovement,
        # tune.grid_search([ExpectedImprovement, RandomAcquisition, GreedyAcquisition]),
        "kappa": 0.5,
        "dropout_proba": 0.4,
        "max_examp_per_epoch": 512,
        # "max_examp_unseen": 2048,
        "acquire_n_at_a_time": 4,
        "n_epoch_between_queries": 3,
        "n_train_forward_passes": 1,
        "n_scoring_forward_passes": 5,
    }

    _, _, summaries_dir = get_external_dirs()
    configuration = {
        "trainer": BasicTrainer,
        "trainer_config": {**pipeline_config, **predictor_config, **model_config, **dataset_config,
                           **active_learning_config},
        "summaries_dir": summaries_dir,
        "memory": 1800,
        "checkpoint_freq": 20,
        "stop": {"training_iteration": 100},
        "checkpoint_at_end": False,
        "resources_per_trial": {},#"cpu": 10, "gpu": 1},
        'asha_metric': "eval_mean",
        'asha_mode': "min",
        'asha_max_t': 100,
        "name": "GraphSignalFirstTrial"
    }

    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=configuration['asha_metric'],
        mode=configuration['asha_mode'],
        max_t=configuration['asha_max_t'],
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )

    search_space = {
        "lr": hp.loguniform("lr", -16.118095651, -5.52146091786),
        "attention_rank": hp.quniform("attention_rank", 50, 300, 1),
        "prot_emb_dim": hp.quniform("prot_emb_dim", 128, 2056, 1),
        "residual_layers_dim": hp.quniform("residual_layers_dim", 64, 1024, 1),
        'num_non_residual_convs': hp.quniform('num_non_residual_convs', 1, 3, 1),
        'num_residual_blocks': hp.quniform('num_residual_blocks', 0, 3, 1),
        'pooling_periodicity': hp.quniform('pooling_periodicity', 1, 3, 1),
        'pooling_ratio': hp.uniform('pooling_ratio', .5, .85),
        'pooling_layer': hp.choice('pooling_layer', [SAGPooling, TopKPooling]),
        'global_pooling': hp.choice('global_pooling', [None, global_max_pool, global_mean_pool, GlobalAttention, Set2Set]),
        'num_pooling_modules': hp.quniform('num_pooling_modules', 0, 5, 1),
    }

    current_best_params = [{
        "lr": 1e-4,
        "attention_rank": 140,
        "prot_emb_dim": 512,
        "residual_layers_dim": 512,
        'num_non_residual_convs': 2,
        'num_residual_blocks': 2,
        'pooling_periodicity': 1,
        'pooling_ratio': .5,
        'pooling_layer': 0,
        'global_pooling': 0,
        'num_pooling_modules': 3,
    }]

    search_alg = HyperOptSearch(
        search_space,
        metric=configuration['asha_metric'],
        mode=configuration['asha_mode'],
        points_to_evaluate=current_best_params
    )

    ####################################################################################################################
    # Use tune
    ####################################################################################################################

    ray.init(num_cpus=40)

    time_to_sleep = 5
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    analysis = tune.run(
        configuration["trainer"],
        name=configuration["name"],
        config=configuration["trainer_config"],
        stop=configuration["stop"],
        resources_per_trial=configuration["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=configuration["checkpoint_at_end"],
        local_dir=configuration["summaries_dir"],
        checkpoint_freq=configuration["checkpoint_freq"],
        scheduler=asha_scheduler,
        search_alg=search_alg,
    )

    ####################################################################################################################
    # Do not use tune
    ####################################################################################################################

    #trainer = configuration["trainer"](configuration["trainer_config"])
    #trainer.train()
