import os.path as osp
from config_model import model_config
from LambdaZero.contrib.model_with_uncertainty import MolMCDropGNN
from LambdaZero.model_with_uncertainty import MolMCDropGNNDeup, MolMCDropGNNGP
import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

acquirer_config = {
    "model": MolMCDropGNN,
    "model_config": model_config,
    "acq_size": 32,
    "kappa": 0.2
}

acquirer_config_deup = {
    "model": MolMCDropGNNDeup,
    "model_config": model_config,
    "acq_size": 32,
    "kappa": 0.2,
}

acquirer_config_gp = {
    "model": MolMCDropGNNGP,
    "model_config": model_config,
    "acq_size": 32,
    "kappa": 0.2,
}

oracle_config = {"num_threads":8,
                 "dockVina_config": {"outpath": osp.join(summaries_dir, "docking")},
                 "mean":-8.6, "std": 1.1,
                 }