import os.path as osp
import numpy as np
import torch
from ray import tune
import LambdaZero.utils
from dataset import MolMaxDist
from layers import DiffGCN

class AlphaSageTrainer(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.dataset = MolMaxDist(config["dataset_mdp_steps"],config["blocks_file"])
        self.conv = config["model"]([14, 1], eps=config["eps"], t=config["t"])
        self.optim = config["optimizer"](self.conv.parameters(), **config["optimizer_config"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv.to(self.device)

    def _train(self):
        losses = []
        for i in range(self.config["dataset_size"]):
            self.optim.zero_grad()
            # for i in range(self.config["b_size"]):
            g = self.dataset()
            g = g.to(self.device)
            x = self.conv(g.x, g.edge_index, g.slices)
            norm_dist = (g.max_dist - 10) / 8.5
            #print("x", x[:,0].shape, "norm_dist", norm_dist.shape)
            loss = ((x[:,0] - norm_dist) ** 2).mean()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            self.optim.step()
        loss_mean = np.mean(losses)
        return {"loss_mean":loss_mean}


# predictive power is almost same as predicting average
# todo: atom embeddings have large numbers and could better be normalized


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
DEFAULT_CONFIG = {
    "regressor_config": {
        "run_or_experiment": AlphaSageTrainer,
        "config": {
            "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
            "dataset_mdp_steps": 0,
            # "b_size": 50,
            "model": DiffGCN,
            "optimizer": torch.optim.Adam,
            "optimizer_config": {"lr": 0.001,},
            "eps": 100,
            "t": 5,
            "dataset_size":1000,
            "device":"cuda",
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 1000
                 },
        "resources_per_trial": {
            "cpu": 4,
            "gpu": 1.0
        },
        "checkpoint_score_attr":"train_loss",
        "num_samples": 1,
        "checkpoint_at_end": False,
    },
    "memory": 10 * 10 ** 9,
}

if __name__ == "__main__":
    config = DEFAULT_CONFIG
    #trainer = AlphaSageTrainer(config["trainer_config"])
    #metrics = trainer._train()
    tune.run(**config["regressor_config"])