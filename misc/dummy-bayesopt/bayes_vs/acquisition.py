
import abc
import typing

import torch
from torch import distributions

from . import bayes_models
from . import querier
from . import chem_ops


class AcquisitionFunc(metaclass=abc.ABCMeta):
    """
    Base class for classes that decide which points to query next.
    """
    @abc.abstractmethod
    def get_batch(self, batch_size=500):
        raise NotImplementedError

    @abc.abstractmethod
    def update_with_seen(self, seen_molecules_smiles, seen_molecules_values):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def create_acquisition_function(cls, queried_results_so_far: typing.List[querier.QueriedHeap], **params):
        """
        This class method creates the acquistion function from a list of all the queried results seen so far.
        :param queried_results_so_far: List of all the queries seen from the different fidelity oracles upto this point.
        In position 0 should be cheap docking. In position -1 should be the querier that this acquisition function is
        seeking to populate.
        """
        raise NotImplementedError


class BestFromPreviousOracleAcq(AcquisitionFunc):
    def __init__(self, previous_molecule_heap: querier.QueriedHeap):
        self.sorted_smiles = previous_molecule_heap.all_smiles_sorted
        self.seen_smiles = set()

    def get_batch(self, batch_size=500):

        batch_results = []
        while len(batch_results) < batch_size:
            new_molecule = next(self.sorted_smiles)
            if new_molecule not in self.seen_smiles:
                batch_results.append(new_molecule)

        return batch_results

    def update_with_seen(self, seen_molecules_smiles, seen_molecules_values):
        self.seen_smiles.update(set(seen_molecules_smiles))

    @classmethod
    def create_acquisition_function(cls, queried_results_so_far: typing.List[querier.QueriedHeap], **params):
        return cls(queried_results_so_far[-2])  # ie init with the heap from the previous stage


class ThompsonSamplingAcq(AcquisitionFunc):
    def __init__(self, bayesian_model, available_molecules: typing.Set[str], seen_molecule2value: dict):
        self.available_molecules = available_molecules
        self.seen_molecule2value = seen_molecule2value
        self.bayesian_model = bayesian_model
        self._train_model()

    def update_with_seen(self, seen_molecules_smiles, seen_molecules_values):
        self.seen_molecule2value.update(dict(zip(seen_molecules_smiles, seen_molecules_values)))
        self.available_molecules.difference_update(set(seen_molecules_smiles))
        self._train_model()

    def get_batch(self, batch_size=500):
        # Thompson sampling!
        all_smiles = list(self.available_molecules)
        mvn: distributions.MultivariateNormal = self.bayesian_model.predict_on_smiles(all_smiles)

        indices_available = torch.arange(mvn.loc.shape[0])
        out_smiles = []
        for _ in range(batch_size):
            sample = mvn.sample_n(1)[0]
            samples_of_interest = sample[indices_available]
            indx = torch.argmin(samples_of_interest)
            out_smiles.append(all_smiles[indices_available[indx].item()])
            indices_available = indices_available[torch.arange(indices_available.shape[0]) != indx]
        return out_smiles

    def _train_model(self):
        seen_smiles, seen_values = zip(*self.seen_molecule2value.items())
        seen_smiles = list(seen_smiles)
        seen_values = list(seen_values)
        seen_values_tensor = torch.tensor(seen_values)[:, None]
        self.bayesian_model.fit_on_smiles(list(seen_smiles), seen_values_tensor)

    @classmethod
    def create_acquisition_function(cls, queried_results_so_far: typing.List[querier.QueriedHeap], **params):

        available_molecules = queried_results_so_far[-2].all_smiles_set
        # ^ the available molecules are given by all the molecules seen by the previous oracle

        seen_molecule2value = queried_results_so_far[-1].smi_to_value_dict
        # ^ the values seen so far are those given by the querier that this acquisition function will suggest molecules
        # tp

        if "feature_dim" not in params:
            params["feature_dim"] = 100

        #todo: probably should make the embedding func a seperate class elsewhere
        def embedding_func(list_of_smiles):
            fingerprints = [torch.tensor(chem_ops.morgan_fp_from_smiles(smi, radius=2, number_bits=params["feature_dim"]),
                                         dtype=torch.float32)
                            for smi in list_of_smiles]
            fingerprints = torch.stack(fingerprints)
            # todo: add the previous oracle results to this vector.
            return fingerprints

        embedding_func.fp_dim = params["feature_dim"]

        bayes_regress = bayes_models.BayesianRegression(embedding_func)
        return cls(bayes_regress, available_molecules, seen_molecule2value)



