import os.path as osp
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch.utils.data import Subset

from ray import tune

from custom_dataloader import DL

class BasicRegressor(tune.Trainable):
    def _setup(self, config):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load dataset
        dataset = config["dataset"](**config["dataset_config"])

        # split dataset
        train_idxs, val_idxs, test_idxs = np.load(config["dataset_split_path"], allow_pickle=True)

        sampler = None

        train_idxs = train_idxs[:-1]

        if config['use_sampler']:
            train_dataset = dataset[torch.tensor(train_idxs)]

            if config['use_tail']:
                energies = torch.tensor([-graph.gridscore.item() for graph in train_dataset], dtype=torch.float64)
            else:
                energies = torch.tensor([-graph.gridscore.item() if -graph.gridscore.item() < 63 else 0 for graph in
                                         train_dataset], dtype=torch.float64)

            energies = energies + energies.min()

            if config["mode"] == "pow":
                energies = energies.pow(config['pow'])
            else:
                energies = torch.where(energies > 0, energies.log(), energies)

            energies_prob = energies / energies.sum()
            sampler = torch.utils.data.sampler.WeightedRandomSampler(energies_prob, len(train_dataset))

        if sampler == None:
            self.train_set = DL(Subset(dataset, train_idxs.tolist()), shuffle=True, batch_size=config["b_size"])
        else:
            self.train_set = DL(Subset(dataset, train_idxs.tolist()), batch_size=config["b_size"], samp=sampler)

        self.val_set = DL(Subset(dataset, val_idxs.tolist()), batch_size=config["b_size"])
        self.test_set = DL(Subset(dataset, test_idxs.tolist()), batch_size=config["b_size"])

        # make model
        self.model = config["model"](**config["model_config"])
        self.model.to(self.device)
        self.optim = config["optimizer"](self.model.parameters(), **config["optimizer_config"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.val_set, self.model,  self.device, self.config)
        # rename to make scope
        train_scores = [("train_" + k, v) for k,v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = osp.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
