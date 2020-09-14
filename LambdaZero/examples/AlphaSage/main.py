import sys, time, os.path as osp
import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.utils import merge_dicts
import LambdaZero.utils
from dataset import MolMaxDist
from layers import DiffGCN
import config

class AlphaSageTrainer(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.dataset = MolMaxDist(config["dataset_mdp_steps"],config["blocks_file"])
        self.conv = config["model"](**config["model_par"])
        self.optim = config["optimizer"](self.conv.parameters(), **config["optimizer_config"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv.to(self.device)

    def _train(self):
        message_losses, policy_losses, losses = [],[],[]
        for i in range(self.config["dataset_size"]):
            self.optim.zero_grad()
            # for i in range(self.config["b_size"]):
            g = self.dataset()
            g = g.to(self.device)
            dist_hat, walk_logp = self.conv(g.x, g.edge_index, g.slices)
            norm_dist = (g.max_dist - 10) / 8.5
            # message loss
            message_loss = ((dist_hat[:,0] - norm_dist) ** 2)
            message_loss_mean = message_loss.mean()
            # policy gradient loss
            policy_loss_mean = (walk_logp * message_loss[:,None].detach()).sum(1).mean()
            message_loss_mean.backward(retain_graph=True)
            policy_loss_mean.backward()
            self.optim.step()
            message_losses.append(message_loss_mean.detach().cpu().numpy())
            policy_losses.append(policy_loss_mean.detach().cpu().numpy())
        return {"loss":np.mean(message_losses), "policy_loss":np.mean(policy_losses)}


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
DEFAULT_CONFIG = {
    "regressor_config": {
        "run_or_experiment": AlphaSageTrainer,
        "config": {
            "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
            "dataset_mdp_steps": 2,
            # "b_size": 50,
            "model": DiffGCN,
            "model_par": {
                "channels":[14,1],
                "eps": 0.1,
                "t": 1,
                "diff_aggr_h":128,
                "diff_aggr_l":1,
                "walk_aggr_h":128,
                "walk_aggr_l":1,
            },
            "optimizer": torch.optim.Adam,
            "optimizer_config": {"lr": 0.001,},
            "dataset_size":1000,
            "device":"cuda",
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 50
                 },
        "resources_per_trial": {
            "cpu": 6,
            "gpu": 1.0
        },
        "checkpoint_score_attr":"train_loss",
        "num_samples": 1,
        "checkpoint_at_end": False,
    },
    "memory": 10 * 10 ** 9,
}

if __name__ == "__main__":
    # todo: atom embeddings have large numbers and could better be normalized

    if len(sys.argv) >= 2: config_name = sys.argv[1]
    else: config_name = "pg001"
    config = getattr(config, config_name)
    config = merge_dicts(DEFAULT_CONFIG, config)
    config["regressor_config"]["name"] = config_name

    # #trainer = AlphaSageTrainer(config["trainer_config"])
    # #metrics = trainer._train()
    ray.init(memory=config["memory"])
    print(config["regressor_config"])

    tune.run(**config["regressor_config"])
