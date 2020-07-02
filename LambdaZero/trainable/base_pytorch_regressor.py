import os
from abc import abstractmethod
from typing import Dict

import torch
from ray.tune import tune

from LambdaZero.loggers.ray_tune_logger import RayTuneLogger


class BasePytorchRegressor(tune.Trainable):

    def setup_logger(self, config):
        self.logger = RayTuneLogger(config=config, log_dir=self.logdir, trial_id=self.trial_id)

    @abstractmethod
    def train_epoch(self, training_dataloader, model, optim, device, config):
        pass

    @abstractmethod
    def eval_epoch(self, validation_dataloader, model, device, config):
        pass

    @abstractmethod
    def get_dataloaders(self, config):
        pass

    @abstractmethod
    def get_model(self, config):
        pass

    def set_optimizer(
        self, model: torch.nn.Module, config: Dict
    ) -> torch.optim.Optimizer:
        """
        Sets the optimizer. Default to Adam.

        Args:
            model (torch.nn.Module): What the optimizer will watch
            config (dict): general configuration. Should contain needed configuration to instantiate optimizer

        Returns:
            optimizer (torch.optim.Optimizer)

        """
        return torch.optim.Adam(model.parameters(), lr=config["lr"])

    def _setup(self, config):
        self.setup_logger(config)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # make model
        self.model = self.get_model(config)
        self.model.to(self.device)
        self.optim = self.set_optimizer(self.model, config)

        self.training_dataloader, self.validation_dataloader, self.test_dataloader = self.get_dataloaders(
            config
        )

    @classmethod
    def _combine_scores(cls, train_scores_dict, validation_scores_dict) -> Dict:
        train_scores = [("train_" + k, v) for k, v in train_scores_dict.items()]
        validation_scores = [
            ("validation_" + k, v) for k, v in validation_scores_dict.items()
        ]
        scores_dict = dict(train_scores + validation_scores)
        return scores_dict

    def _train(self):
        train_scores = self.train_epoch(
            self.training_dataloader, self.model, self.optim, self.device, self.config
        )
        validation_scores = self.eval_epoch(
            self.validation_dataloader, self.model, self.device, self.config
        )

        scores = self._combine_scores(train_scores, validation_scores)
        return scores

    def _log_result(self, result):
        res_dict = {
            str(k): v
            for k, v in result.items()
            if (v and "config" not in k and not isinstance(v, str))
        }
        step = result["training_iteration"]
        self.logger.log_metrics(res_dict, step=step)

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
