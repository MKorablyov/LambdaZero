import argparse
import os, time
import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import (train, test, get_data_loaders, ConvNet)


from ray.tune.stopper import Stopper, NoopStopper



class EarlyStop(Stopper):
    def __init__(self):
        self.acc = []

    def __call__(self, trial_id, result):

        self.acc.append(result["mean_acc"])

        if result["training_iteration"] > 1000:
            return True
        else:
            return False

    def stop_all(self):
        return False



class TrainMNIST(tune.Trainable):
    def _setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))

    def _train(self):
        train(self.model, self.optimizer, self.train_loader, device=self.device)
        accc = test(self.model, self.test_loader, self.device)
        return {"mean_acc": accc}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


# class TrainMNIST2(tune.Trainable):
#     def _setup(self, config):
#
#         use_cuda = config.get("use_gpu") and torch.cuda.is_available()
#         self.device = torch.device("cuda" if use_cuda else "cpu")
#         self.train_loader, self.test_loader = get_data_loaders()
#         self.model = ConvNet().to(self.device)
#         #self.optimizer = optim.SGD(
#         #    self.model.parameters(),
#         #    lr=config.get("lr", 0.01),
#         #    momentum=config.get("momentum", 0.9))
#
#     def _train(self):
#         #train(self.model, self.optimizer, self.train_loader, device=self.device)
#         #acc = test(self.model, self.test_loader, self.device)
#
#         analysis = tune.run(
#             TrainMNIST,
#             stop=EarlyStop(),
#             resources_per_trial={
#                 "cpu": 1,
#                 "gpu": 0.1
#             },
#             num_samples=1,
#             checkpoint_at_end=False,
#             checkpoint_freq=100000)
#
#         return {"mean_accuracy": acc}
#
#     def _save(self, checkpoint_dir):
#         checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
#         torch.save(self.model.state_dict(), checkpoint_path)
#         return checkpoint_path
#
#     def _restore(self, checkpoint_path):
#         self.model.load_state_dict(torch.load(checkpoint_path))






if __name__ == "__main__":
    ray.init()
    from LambdaZero.utils import get_external_dirs
    _, _, summaries_dir = get_external_dirs()
    analysis = tune.run(
        TrainMNIST,
        stop=EarlyStop(),
        resources_per_trial={
            "cpu": 1,
            "gpu": 0.1
        },
        num_samples=1,
        checkpoint_at_end=False,
        local_dir=summaries_dir,

        checkpoint_freq=100000)


    print("Best config is:", analysis.get_best_config(metric="mean_acc"))