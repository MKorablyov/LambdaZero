
import abc
import typing

import torch
from torch import nn
from torch import optim

import numpy as np
from tqdm import tqdm

from . import chem_ops


class Oracle(metaclass=abc.ABCMeta):
    def query(self, list_of_smiles: typing.List[str]) -> typing.List[float]:
        return self._query_smiles_batch(list_of_smiles)

    @abc.abstractmethod
    def _query_smiles_batch(self, list_of_smiles: typing.List[str]):
        raise NotImplementedError


class PreloadedOracle(Oracle):
    def __init__(self, initial_dict):
        super().__init__()
        self._properties = initial_dict

    def _query_smiles_batch(self, list_of_smiles: typing.List[str]):
        return [self._properties[smi] for smi in list_of_smiles]


class StructuredNoiseOracle(nn.Module, Oracle):
    """
    This class tries to _mimic_ the noise we might see from Docking, FEP, so it returns the true oracles value with
    noise added.
    The noise added is heterogeneous, and is given by a series of noise networks.
    """
    def __init__(self, true_oracle: Oracle, num_networks=10, fp_dim=500):
        """
        :param true_oracle:
        :param num_networks:
        :param fp_dim:
        """
        super().__init__()

        self.true_oracle = true_oracle

        # Set up some noise networks -- with random initialisation that will not then change
        hidden_layer_size = 128
        self.fp_dim = fp_dim
        self.noise_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(fp_dim, hidden_layer_size), nn.ReLU(), nn.Linear(hidden_layer_size, 1))
             for _ in range(num_networks)])
        for param in self.parameters():
            param.requires_grad = False  # these noise networks we never want to change

        # Each of these noise values may get translated and scaled by a different amount:
        self.scaling_factors = nn.Parameter(torch.randn(num_networks)[None, :], requires_grad=True)  # [1, num_networks]
        self.scaling_mns = nn.Parameter(torch.zeros(num_networks)[None, :], requires_grad=True)  # [1, num_networks]

    def _query_smiles_batch(self, list_of_smiles: typing.List[str]):
        noise_to_add = self.noise(list_of_smiles).tolist()  # [b]
        original_values = self.true_oracle.query(list_of_smiles) # [b]

        out = list(map(sum, zip(original_values, noise_to_add)))
        return out

    def noise(self, list_of_smiles: typing.List[str]) -> torch.Tensor:
        fingerprints = torch.stack([torch.tensor(
            chem_ops.morgan_fp_from_smiles(smi, radius=4, number_bits=self.fp_dim), dtype=torch.float32)
                                    for smi in list_of_smiles
                                    ]).to(self._device)  # [b, fp_dim]
        noise_to_add = torch.cat([noise_net(fingerprints) for noise_net in self.noise_list], dim=1)  # [b, num_networks]
        noise_to_add = (noise_to_add + self.scaling_mns) * self.scaling_factors
        noise_to_add = noise_to_add.sum(dim=1)  # [b]
        return noise_to_add

    @property
    def _device(self):
        return next(self.parameters()).device

    def fit_scales(self, mean_to_match, var_to_match,
                   list_of_possible_smiles: typing.List[str],
                   rng: np.random.RandomState, batch_size=1000, niter=150, lr=0.01):

        # We first get the scales roughly right by scaling up the factors to match the standard deviation on one initial
        # batch.
        with torch.no_grad():
            smiles_this_batch = rng.choice(list_of_possible_smiles, size=5000, replace=False).tolist()
            noise = self.noise(smiles_this_batch)
            self.scaling_factors[...] = self.scaling_factors / torch.std(noise) * np.sqrt(var_to_match)

        # We then tune the top parameters using gradient descent:
        # we could just follow out step above for the means, but left it using gradient descent as initially we used
        # MMD to match distributions (in case we wanted to match more complicated distributions) and as it does not
        # take too long to run anyway decided to leave this as it is.
        optimizer = optim.Adam(iter([self.scaling_factors, self.scaling_mns]), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        desc = "l: {:.3f} ({:.3f},{:.3f})"
        pbar = tqdm(initial=0, leave=False, total=niter, desc=desc.format(0, 0, 0))

        for i in tqdm(range(niter)):
            smiles_this_batch = rng.choice(list_of_possible_smiles, size=batch_size, replace=False).tolist()

            optimizer.zero_grad()
            noise = self.noise(smiles_this_batch)

            loss_mean = (torch.mean(noise) - mean_to_match)**2
            loss_variance = (torch.var(noise) - var_to_match)**2
            loss = loss_mean + loss_variance
            # ^ nb our loss is going to just be the sum of the squared difference to our desired mean and variance with
            # the current ones.

            loss.backward()

            optimizer.step()
            scheduler.step()

            pbar.desc = desc.format(loss.item(), loss_mean.item(), loss_variance.item())
            pbar.update()

            if i % 50 == 0:
                tqdm.write(f"Scaling factors: {self.scaling_factors},\n Scaling bias: {self.scaling_mns}")

        pbar.close()











