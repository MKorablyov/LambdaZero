"""
This script creates the mock oracles for our dummy task.

We treat the dockscores from Dopamine as the ground truth. We then add structured noise (from randomly initialised NNs)
to this and tune the final scale/bias terms of these NNs so that the final variance matches what we expect the variance
in the errors of the different molecules to be.

"""

import pyarrow.parquet as pq
import torch
import numpy as np

from bayes_vs import dummy_oracles


class Params:
    def __init__(self):
        # Below we store a list of the oracles we want to mock up. For each we have a name, a target variance for the
        # error distribution and a learning rate for tuning the op of the randomly initialized NNs to reach this target
        # variance.
        self.dummy_oracle_name_variance_lr_tuples = [("cheap-docking", 2.7, 0.05),
                                             ("expensive-docking", 2.5, 0.05),
                                             ("FEP", 0.8, 0.02)]
        # ^ see Maksym's note for where these variance com
        self.smiles2grndtruth = self._get_ground_truth()

    def _get_ground_truth(self):
        path = '../../brutal_dock/d4/raw/d4_250k.parquet'
        d4_250k = pq.read_table(path).to_pandas()

        dockscore = d4_250k['dockscore']
        dockscore = dockscore - dockscore.mean()
        dockscore = dockscore / 4.2
        # ^ divide through by 4.2 -- see Maksym's note this is to normalize Dockscores so they lie in the same range as
        # we expect the FEP scores to be in.

        out = dict(zip(d4_250k['smiles'], dockscore))
        return out


def main(params: Params):
    ground_truth_oracle = dummy_oracles.PreloadedOracle(params.smiles2grndtruth)
    rng = np.random.RandomState(8526)
    all_possible_smiles = list(params.smiles2grndtruth.keys())

    save_out = {'ground-truth': params.smiles2grndtruth}

    for oracle_name, variance_to_target, lr in params.dummy_oracle_name_variance_lr_tuples:
        print(f"Working on: {oracle_name}")
        sim_oracle = dummy_oracles.StructuredNoiseOracle(ground_truth_oracle)
        sim_oracle.fit_scales(0., variance_to_target, all_possible_smiles, rng, niter=100, lr=lr)
        save_out[f"{oracle_name}_state_dict"] = sim_oracle.state_dict()

    torch.save(save_out, "trained_oracles.chkpt")


if __name__ == '__main__':
    main(Params())

