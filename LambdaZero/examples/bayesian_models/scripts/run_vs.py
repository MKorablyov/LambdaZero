
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

@dataclass
class VSExperiment:
    name: str
    acquisition_strategies: typing.Dict[str, typing.List[AcquisitionStrategy]]


class Params:
    def __init__(self):
        self.oracles_chkpt_name = 'trained_oracles.chkpt'
        self.op_name = "vs_results.pick"

        self.experiments = [
            VSExperiment("PlainVS",
                         {
                             "expensive-docking":
                             [
                                 AcquisitionStrategy(50, 500, acquisition.BestFromPreviousOracleAcq)],
                             "FEP":
                             [
                                 AcquisitionStrategy(15, 20, acquisition.BestFromPreviousOracleAcq)
                             ],
                             "gt":
                                 [
                                     AcquisitionStrategy(5, 10, acquisition.BestFromPreviousOracleAcq),
                                 ]
                         }),
            VSExperiment("BayesVS",
                         {
                             "expensive-docking":
                                 [
                                     AcquisitionStrategy(50, 500, acquisition.BestFromPreviousOracleAcq)],
                             "FEP":
                                 [
                                     AcquisitionStrategy(5, 20, acquisition.BestFromPreviousOracleAcq),
                                     AcquisitionStrategy(10, 20, acquisition.ThompsonSamplingAcq)
                                 ],
                             "gt":
                                 [
                                     AcquisitionStrategy(5, 10, acquisition.BestFromPreviousOracleAcq),
                                 ]
                         }),
            ]


def main(params: Params):

    # Set up oracles
    chkpt = torch.load(params.oracles_chkpt_name)
    oracles = collections.OrderedDict()
    oracles['gt'] = dummy_oracles.PreloadedOracle(chkpt['ground-truth'])

    for name in ['cheap-docking_state_dict', 'expensive-docking_state_dict', 'FEP_state_dict']:
        state_dict = chkpt[name]
        model_name = name.split('_')[0]
        oracle = dummy_oracles.StructuredNoiseOracle(oracles['gt'], num_networks=10)
        oracle.load_state_dict(state_dict)
        oracles[model_name] = oracle
    all_molecules = list(chkpt['ground-truth'].keys())
    # all_molecules = list(chkpt['ground-truth'].keys())[:50000]

    experiments = {}


    for experiment in params.experiments:
        print(f"\n# Running {experiment.name}")
        print(f"\n this follows {experiment}")

        # Query all of the cheap docking oracles points
        experiment_queriers = [querier.QueriedHeap()]
        num_batches = 1000
        for batch in tqdm(np.array_split(all_molecules, num_batches), desc="Querying cheap docking"):
            batch = batch.tolist()
            results = oracles['cheap-docking'].query(batch)
            for smi, res in zip(batch, results):
                experiment_queriers[-1].add_result(querier.OracleResult(res, smi))

        # Now we will run through expensive docking, cheap docking and ground truth
        stages = ['expensive-docking', 'FEP', 'gt']
        for stage_name in stages:
            experiment_queriers.append(querier.QueriedHeap())
            oracle_ = oracles[stage_name]

            # For each of these we may have a series of acquisition strategies (eg start with best from prev before
            # switching to Bayes Opt)
            for i, acq_strategy in enumerate(experiment.acquisition_strategies[stage_name]):
                acq_func = acq_strategy.acquistion.create_acquisition_function(experiment_queriers)

                # Finally for each acquisition strategy we may have a series of batches to run through.
                for i in tqdm(range(acq_strategy.num_batches), desc=f"{stage_name}, acq{i}"):
                    smiles_to_query = acq_func.get_batch(acq_strategy.batch_size)
                    results = oracle_.query(smiles_to_query)
                    for smi, res in zip(smiles_to_query, results):
                        experiment_queriers[-1].add_result(querier.OracleResult(res, smi))
                    acq_func.update_with_seen(smiles_to_query, results)

        experiments[experiment.name] = dict(
            experiment_queriers=experiment_queriers, stages=['cheap-docking'] + stages
        )

    # todo: all these nested for loops are very messy so break out

    with open(params.op_name, 'wb') as fo:
        pickle.dump(
            dict(oracles=oracles, experiments=experiments),
            fo)


if __name__ == '__main__':
    main(Params())
