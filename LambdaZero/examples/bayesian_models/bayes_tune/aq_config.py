import os.path as osp
import os
import numpy as np
from ray.tune import grid_search
import LambdaZero.utils
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mcdrop, train_mpnn_brr, \
    mpnn_brr_mean_variance
    
datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()
small_path =  osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy")
big_path = osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k.npy")

small_file_name = ["Zinc15_2k"]
big_file_name = ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]

uct000 = {}
                                           
# this will apply mpnn_blr uncertainty model instead of mcdrop
uct001 = {
    "acquirer_config":{
        "config":{
            "regressor_config":{
                "config":
                    {"train": train_mpnn_brr,
                     "get_mean_variance": mpnn_brr_mean_variance,
                     "model_config": {"do_dropout" : False, "dropout_in_data" : False, "drop_etc" : False},
                     "dataset_split_path": small_path,
                     "dataset_config": {
                        "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                        "props": ["gridscore", "smi"],
                        "file_names": small_file_name}
                }
            }
        }
    }
}


# todo
# (1) mpnn bll + UCT + find kappa on small and large datasets
# small: python example_UCT uct002
# large: python example_UCT uct003
uct002 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":grid_search(list(np.linspace(start=0.01,stop=100,num=20))),
                #"model_config": {"do_dropout": False, "dropout_in_data": False, "drop_etc": False},
                "dataset_split_path": small_path,
                "dataset_config": {
                    "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                    "props": ["gridscore", "smi"],
                    "file_names": small_file_name},
                "regressor_config":{
                    "config":{
                        "model_config": {"do_dropout" : False, "dropout_in_data" : False, "drop_etc" : False}
                    }
                }
        }
    }
}

uct003 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":grid_search(list(np.linspace(start=0.01,stop=100,num=20))),
                "dataset_split_path": big_path,
                "dataset_config": {
                    "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                    "props": ["gridscore", "smi"],
                    "file_names": big_file_name},
                "regressor_config":{
                    "config":{
                        "model_config": {"do_dropout" : False, "dropout_in_data" : False, "drop_etc" : False}
                    }
                }
        }
    }
}

# # # (2) with the best kappa try different amounts of noise
# # # baseline without noise, random (epislon = 1000), greedy/egreedy (epislon between 0 and 1)
# # small:  python example_UCT uct004
# # large:  python example_UCT uct005
uct004 = {
    "acquirer_config":{
        "config":{
            "epsilon": grid_search([0, 1000]),
          # "kappa": using the best kappa found
            "dataset_split_path": small_path,
                "dataset_config": {
                    "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                    "props": ["gridscore", "smi"],
                    "file_names": small_file_name},
                "regressor_config":{
                    "config":{
                        "model_config": {"do_dropout" : False, "dropout_in_data" : False, "drop_etc" : False}
                    }
                }
        }
    }
}

# # # (3) find the best alpha for mpnn + mcdrop
# # # see config.py

