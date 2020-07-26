
import abc
import typing

import torch
from torch import distributions

<<<<<<< HEAD:misc/dummy-bayesopt/bayes_vs/acquisition.py
import greedy_models
import bayes_models
import querier
import chem_ops

=======
from . import greedy_models
from . import bayes_models
from . import querier
from . import chem_ops
>>>>>>> master:LambdaZero/examples/bayesian_models/bayes_vs/acquisition.py


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
    # previous_molecule_heap.all_smiles_sorted
    # sort them and grab the first from sorted bc next(self.sorted_smiles)
    def __init__(self, previous_molecule_heap: querier.QueriedHeap):
        self.sorted_smiles = previous_molecule_heap.all_smiles_sorted
        self.seen_smiles = set() #create a empty set

    def get_batch(self, batch_size=500): # get 500 unseen mole

        batch_results = []
        while len(batch_results) < batch_size:
            new_molecule = next(self.sorted_smiles)
            if new_molecule not in self.seen_smiles:
                batch_results.append(new_molecule)

        return batch_results

    def update_with_seen(self, seen_molecules_smiles, seen_molecules_values): #grab the new seen mole, update the seen_smiles list
        self.seen_smiles.update(set(seen_molecules_smiles))

    @classmethod #**params: other parameters; classmethod: a fufnction return class
    def create_acquisition_function(cls, queried_results_so_far: typing.List[querier.QueriedHeap], **params):
        return cls(queried_results_so_far[-2])  # ie init with the heap from the previous stage
#cls:same class. pass queried_results_so_far as input para; [-2]- all available/unseen molecule, take it as acquisition

class ThompsonSamplingAcq(AcquisitionFunc):
    #available_molecules:unseen; seen_molecule2value: seen w/ mole+dockscore (x,y) dictionary
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

        previous_molecule_to_result_dicts = [q.smi_to_value_dict for q in queried_results_so_far[:-1]]

        #todo: probably should make the embedding func a seperate class elsewhere
        def embedding_func(list_of_smiles):

            previous_oracle_results = [[d[smi] for d in previous_molecule_to_result_dicts] for smi in list_of_smiles]
            previous_oracle_results = torch.tensor(previous_oracle_results, dtype=torch.float32)

            fingerprints = [torch.tensor(chem_ops.morgan_fp_from_smiles(smi, radius=2, number_bits=params["feature_dim"]),
                                         dtype=torch.float32)
                            for smi in list_of_smiles]
            #todo: representations.
            fingerprints = torch.stack(fingerprints)

            feats = torch.cat([previous_oracle_results, fingerprints], dim=1)
            #todo: maybe NN on top

            return feats

        embedding_func.fp_dim = params["feature_dim"] + len(previous_molecule_to_result_dicts)

        bayes_regress = bayes_models.BayesianRegression(embedding_func)
        return cls(bayes_regress, available_molecules, seen_molecule2value)

class GreedySamplingAcq(AcquisitionFunc):
    #own trainer, use John's available_molecules, seen_molecule2value
    def __init__(self, trainer, available_molecules: typing.Set[str], seen_molecule2value: dict, noise_std = 1., noise_mean = 0.):
        self.available_molecules = available_molecules
        self.seen_molecule2value = seen_molecule2value
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.trainer = trainer
        self._train_model()
#abc
    def update_with_seen(self, seen_molecules_smiles, seen_molecules_values):
        self.seen_molecule2value.update(dict(zip(seen_molecules_smiles, seen_molecules_values)))
        self.available_molecules.difference_update(set(seen_molecules_smiles)) # take avail mo subtract, rm from list
        #avil 100k, seen2value 500, so avil 100k-500, seen2value 1000
        #update - append to seen2value list
        self._train_model()

#abc
    def get_batch(self, batch_size=500): #get unseen 500
        fingerprints = [torch.tensor(chem_ops.morgan_fp_from_smiles(smi, radius=2, number_bits=1024),
                                         dtype=torch.float32)
                            for smi in self.available_molecules] #available_molecules in smiles format, convert smi to fp 1024 encoding
        fingerprints = torch.stack(fingerprints) #stack and pass to NN
        predicted_scores = self.trainer.predict(fingerprints) 
        predicted_scores = predicted_scores * self.noise_std + self.noise_mean # add uncertainty

        sorted_scores = sorted(predicted_scores)
        sorted_score_index = [index for index, num in sorted(enumerate(predicted_scores), key=lambda x: x[-1])] #reverse sorting
        #lambda is inline fct (key want to follow) take idx of sorted score, 
        #x input, x[-1] (highest, best) output
        #give index of best out all avilable molecule - close to 100k (x_out)

        #create a idx list
        #look at pred score, make a idx of their idx, so we will be able to look at avilable mole and create a
        #5,1,8 (5 is the highest, 1 is the second highest) - sort all the mo together

        sorted_available_molecules = list(self.available_molecules) 
        sorted_available_molecules = set([sorted_available_molecules[i] for i in sorted_score_index]) #100k-length(seen)
        #reorder the avil mole in order of sorted_score idx

        sorted_available_molecules = iter(sorted_available_molecules) #creates a iterator, iterate the set 
        seen_smiles, seen_values = zip(*self.seen_molecule2value.items())#smiles and dockscore
        batch_results = []

#grab batch_size unseen molecule
        while len(batch_results) < batch_size:
            new_molecule = next(sorted_available_molecules)
            if new_molecule not in list(seen_smiles):
                batch_results.append(new_molecule)

        return batch_results

    def _train_model(self):
        seen_smiles, seen_values = zip(*self.seen_molecule2value.items())
        seen_smiles = list(seen_smiles)
        seen_values = list(seen_values)
        fingerprints = [torch.tensor(chem_ops.morgan_fp_from_smiles(smi, radius=2, number_bits=1024),
                                         dtype=torch.float32)
                            for smi in seen_smiles]
        seen_values_tensor = torch.tensor(seen_values)[:, None]
        fingerprints = torch.stack(fingerprints)
        self.trainer.train(fingerprints, seen_values_tensor, batch_size = 500)
#abc
    @classmethod
    def create_acquisition_function(cls, queried_results_so_far: typing.List[querier.QueriedHeap], **params):

        available_molecules = queried_results_so_far[-2].all_smiles_set
        # ^ the available molecules are given by all the molecules seen by the previous oracle

        seen_molecule2value = queried_results_so_far[-1].smi_to_value_dict
        # ^ the values seen so far are those given by the querier that this acquisition function will suggest molecules
        # tp

<<<<<<< HEAD:misc/dummy-bayesopt/bayes_vs/acquisition.py
class GreedySamplingAcq(AcquisitionFunc):
    #own trainer, use John's available_molecules, seen_molecule2value
    def __init__(self, trainer, available_molecules: typing.Set[str], seen_molecule2value: dict, noise_std = 1., noise_mean = 0.):
        self.available_molecules = available_molecules
        self.seen_molecule2value = seen_molecule2value
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.trainer = trainer
        self._train_model()
#abc
    def update_with_seen(self, seen_molecules_smiles, seen_molecules_values):
        self.seen_molecule2value.update(dict(zip(seen_molecules_smiles, seen_molecules_values)))
        self.available_molecules.difference_update(set(seen_molecules_smiles)) # take avail mo subtract, rm from list
        #avil 100k, seen2value 500, so avil 100k-500, seen2value 1000
        #update - append to seen2value list
        self._train_model()

#abc
    def get_batch(self, batch_size=500): #get unseen 500
        fingerprints = [torch.tensor(chem_ops.morgan_fp_from_smiles(smi, radius=2, number_bits=1024),
                                         dtype=torch.float32)
                            for smi in self.available_molecules] #available_molecules in smiles format, convert smi to fp 1024 encoding
        fingerprints = torch.stack(fingerprints) #stack and pass to NN
        predicted_scores = self.trainer.predict(fingerprints) 
        predicted_scores = predicted_scores * self.noise_std + self.noise_mean # add uncertainty

        sorted_scores = sorted(predicted_scores)
        sorted_score_index = [index for index, num in sorted(enumerate(predicted_scores), key=lambda x: x[-1])] #reverse sorting
        #lambda is inline fct (key want to follow) take idx of sorted score, 
        #x input, x[-1] (highest, best) output
        #give index of best out all avilable molecule - close to 100k (x_out)

        #create a idx list
        #look at pred score, make a idx of their idx, so we will be able to look at avilable mole and create a
        #5,1,8 (5 is the highest, 1 is the second highest) - sort all the mo together

        sorted_available_molecules = list(self.available_molecules) 
        sorted_available_molecules = set([sorted_available_molecules[i] for i in sorted_score_index]) #100k-length(seen)
        #reorder the avil mole in order of sorted_score idx

        sorted_available_molecules = iter(sorted_available_molecules) #creates a iterator, iterate the set 
        seen_smiles, seen_values = zip(*self.seen_molecule2value.items())#smiles and dockscore
        batch_results = []

#grab batch_size unseen molecule
        while len(batch_results) < batch_size:
            new_molecule = next(sorted_available_molecules)
            if new_molecule not in list(seen_smiles):
                batch_results.append(new_molecule)

        return batch_results

    def _train_model(self):
        seen_smiles, seen_values = zip(*self.seen_molecule2value.items())
        seen_smiles = list(seen_smiles)
        seen_values = list(seen_values)
        fingerprints = [torch.tensor(chem_ops.morgan_fp_from_smiles(smi, radius=2, number_bits=1024),
                                         dtype=torch.float32)
                            for smi in seen_smiles]
        seen_values_tensor = torch.tensor(seen_values)[:, None]
        fingerprints = torch.stack(fingerprints)
        self.trainer.train(fingerprints, seen_values_tensor, batch_size = 500)
#abc
    @classmethod
    def create_acquisition_function(cls, queried_results_so_far: typing.List[querier.QueriedHeap], **params):

        available_molecules = queried_results_so_far[-2].all_smiles_set
        # ^ the available molecules are given by all the molecules seen by the previous oracle

        seen_molecule2value = queried_results_so_far[-1].smi_to_value_dict
        # ^ the values seen so far are those given by the querier that this acquisition function will suggest molecules
        # tp

=======
>>>>>>> master:LambdaZero/examples/bayesian_models/bayes_vs/acquisition.py
        # bayes_regress = bayes_models.BayesianRegression(embedding_func)
        #TODO: implement proper init for model
        model = greedy_models.Model()
        trainer = greedy_models.Trainer(model)
        return cls(trainer, available_molecules, seen_molecule2value)