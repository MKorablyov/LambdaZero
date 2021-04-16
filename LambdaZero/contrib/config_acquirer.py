import os.path as osp
from LambdaZero.contrib.config_model import model_config
from LambdaZero.contrib.model_with_uncertainty import MolMCDropGNN
import LambdaZero.utils
from LambdaZero.contrib.functional import elu2
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

# Candidate Molecule hard thresholds for logs
# Should be fixed in order to be able to compare runs across different training configs
LOG_CANDIDATE_QED_SCORE = 0.5
LOG_CANDIDATE_SYNTH_SCORE = 4.

acquirer_config = {
    "model": MolMCDropGNN,
    "model_config": model_config,
    "acq_size": 32,
    "kappa": 0.2,
}

oracle_config = {"num_threads":8,
                 "dockVina_config": {"outpath": osp.join(summaries_dir, "docking")},
                 "mean":-8.6, "std": 1.1, "act_y":elu2,
                 }