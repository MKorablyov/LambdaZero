"""
Simple Bayesian Optimization (BO) runs just one level of querying. Ie we run BO to try to discover the best molecules
according to cheap docking score, assuming we have none.
"""

import typing
import collections
import pickle
from dataclasses import dataclass

import torch
import numpy as np

from tqdm import tqdm

from bayes_vs import dummy_oracles
from bayes_vs import acquisition
from bayes_vs import querier


@dataclass
class AcquisitionStrategy:
    num_batches: int
    batch_size: int
    acquistion: typing.Type[acquisition.AcquisitionFunc]
    acquistion_params: dict

@dataclass
class VSExperiment:
    name: str
    acquisition_strategies: typing.List[AcquisitionStrategy]


class Params:
    def __init__(self):
        self.oracles_chkpt_name = 'trained_oracles_13May2020.chkpt'
        self.op_name = "simple_bo_results.pick"

        self.rng = np.random.RandomState(56)

        self.experiments = [
            VSExperiment("Random",
                             [
                                 AcquisitionStrategy(50, 100, acquisition.BestFromPreviousOracleAcq, {'prev_oracle_as_feats_flag': False}),
                             ]),
            VSExperiment("Thompson",
                             [
                                 AcquisitionStrategy(1, 100, acquisition.BestFromPreviousOracleAcq,
                                                     {'prev_oracle_as_feats_flag': False}),
                                 AcquisitionStrategy(49, 100, acquisition.ThompsonSamplingAcq, {'prev_oracle_as_feats_flag': False}),
                             ]),
            ]


def main(params: Params):
    # Set up ground truth  -- in this example it is just the cheap docking score from the fake task
    chkpt = torch.load(params.oracles_chkpt_name)
    oracles = collections.OrderedDict()
    oracles['gt'] = dummy_oracles.PreloadedOracle(chkpt['ground-truth'])

    all_molecules = list(chkpt['ground-truth'].keys())

    experiments = {}
    for experiment in params.experiments:
        print(f"\n# Running {experiment.name}")
        print(f"\n this follows {experiment}")

        # We are first going to add to all of the molecules a random number -- this will allow us to hack this task
        # into the general code we wrote for before for multistage BO. As we can use best from previous oracle
        # with this before to get a random oracle!
        experiment_queriers = [querier.QueriedHeap()]
        random_scores = params.rng.randn(len(all_molecules)).tolist()
        for smi, res in zip(all_molecules, random_scores):
            experiment_queriers[-1].add_result(querier.OracleResult(res, smi))

        # Then we shall do the proper run where we actually query scores
        experiment_queriers.append(querier.QueriedHeap())
        oracle_ = oracles['gt']

        for i, acq_strategy in enumerate(experiment.acquisition_strategies):
            acq_func = acq_strategy.acquistion.create_acquisition_function(experiment_queriers,
                                                                           **acq_strategy.acquistion_params)

            # Finally for each acquisition strategy we may have a series of batches to run through.
            for i in tqdm(range(acq_strategy.num_batches), desc=f"acq{i}"):
                smiles_to_query = acq_func.get_batch(acq_strategy.batch_size)
                results = oracle_.query(smiles_to_query)
                for smi, res in zip(smiles_to_query, results):
                    experiment_queriers[-1].add_result(querier.OracleResult(res, smi))
                acq_func.update_with_seen(smiles_to_query, results)

        # Then put the results into the results dict
        experiments[experiment.name] = dict(
            experiment_queriers=experiment_queriers
        )

    with open(params.op_name, 'wb') as fo:
        pickle.dump(
            dict(oracles=oracles, experiments=experiments),
            fo)

if __name__ == '__main__':
    main(Params())
    print("done!")
