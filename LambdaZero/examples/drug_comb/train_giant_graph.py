import torch
from LambdaZero.examples.drug_comb.dataset.drug_combdb_data import DrugCombDb
from LambdaZero.examples.drug_comb.dataset.drug_response_data import DrugResponse
from LambdaZero.examples.drug_comb.model.multi_message_gcn import GiantGraphMPNN
from LambdaZero.examples.drug_comb.model.baseline import BaselineMLP, SimpleBaselineFP, BaselineProt, \
    SimpleBaselineProt, Dummy
from LambdaZero.examples.drug_comb.utils.utils import random_split
import os
from LambdaZero.utils import get_external_dirs
from torch.utils.data import TensorDataset, DataLoader
import time
from ray import tune
import ray


def train_epoch(data, loader, model, optim):
    model.train()
    epoch_loss = 0
    num_batches = len(loader)

    for i, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch)
        loss = model.loss(out, drug_drug_batch)

        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    print('Mean train loss: {:.4f}'.format(epoch_loss / num_batches))

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches}


def eval_epoch(data, loader, model):
    model.eval()
    epoch_loss = 0
    num_batches = len(loader)

    with torch.no_grad():
        for i, drug_drug_batch in enumerate(loader):
            out = model.forward(data, drug_drug_batch)

            loss = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item()

    print('Mean valid loss: {:.4f}'.format(epoch_loss / num_batches))

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches}


########################################################################################################################
# Main training loop
########################################################################################################################


class GiantGraphTrainer(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = config["dataset"](transform=config["transform"], pre_transform=config["pre_transform"],
                                    scores=config["scores"], cell_line=config['cell_line'])

        self.data = dataset[0].to(self.device)

        train_idxs, val_idxs, test_idxs = dataset.random_split(config["test_set_prop"], config["val_set_prop"])

        # Train loader
        train_ddi_dataset = TensorDataset(self.data.ddi_edge_idx[:, train_idxs].T,
                                          self.data.ddi_edge_classes[train_idxs], self.data.ddi_edge_attr[train_idxs])

        self.train_loader = DataLoader(train_ddi_dataset,
                                       batch_size=config["batch_size"],
                                       pin_memory=(self.device == 'cpu'))

        # Valid loader
        valid_ddi_dataset = TensorDataset(self.data.ddi_edge_idx[:, val_idxs].T,
                                          self.data.ddi_edge_classes[val_idxs], self.data.ddi_edge_attr[val_idxs])

        self.valid_loader = DataLoader(valid_ddi_dataset, batch_size=config["batch_size"],
                                       pin_memory=(self.device == 'cpu'))

        config["device"] = self.device

        # Initialize model and optimizer
        self.model = config["model"](config, self.data).to(self.device)
        # self.model = GCN(config).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.data, self.train_loader, self.model, self.optim)
        eval_scores = self.eval_epoch(self.data, self.valid_loader, self.model)

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


if __name__ == '__main__':
    ray.init()

    time_to_sleep = 30
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    _, _, summaries_dir = get_external_dirs()
    configuration = {
        "trainer": GiantGraphTrainer,
        "trainer_config": {
            "transform": None,
            "pre_transform": None,
            "scores": ['HSA'],  # tune.grid_search([['ZIP'], ['Bliss'], ['Loewe'], ['HSA']]),
            "layers": [1024, 1],  # tune.grid_search([[1024, 512], [1024, 100], [1024, 50], [1024, 20], [1024, 5]]),
            "val_set_prop": 0.2,
            "test_set_prop": 0.0,
            "lr": 1e-4,  # tune.grid_search([1e-4, 1e-5]),
            "model": BaselineProt,  # tune.grid_search([GiantGraphMPNN, BaselineMLP]),
            "dataset": DrugCombDb,
            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
            "embed_channels": 256,
            "num_epochs": 256,
            'cell_line': 'one',
            "batch_size": 512,
        },
        "summaries_dir": summaries_dir,
        "memory": 20 * 10 ** 9,
        "checkpoint_freq": 200,
        "stop": {"training_iteration": 600},
        "checkpoint_at_end": False,
        "resources_per_trial": {"gpu": 1},
        "name": "SimpleBaselineProtOneCellLine"
    }

    analysis = tune.run(
        configuration["trainer"],
        name=configuration["name"],
        config=configuration["trainer_config"],
        stop=configuration["stop"],
        resources_per_trial=configuration["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=configuration["checkpoint_at_end"],
        local_dir=configuration["summaries_dir"],
        checkpoint_freq=configuration["checkpoint_freq"]
    )

    # trainer = GiantGraphTrainer(configuration["trainer_config"])
    # trainer.train()
