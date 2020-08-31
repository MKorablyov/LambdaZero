import os.path as osp
import os
import numpy as np
import torch
from ray.tune import grid_search
from torch_geometric import transforms as T
import LambdaZero.utils
import LambdaZero.inputs
import LambdaZero.models
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mcdrop, train_mpnn_brr, \
    mpnn_brr_mean_variance
from LambdaZero.examples.bayesian_models.bayes_tune.brr import BRR
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_epoch,eval_epoch, train_mcdrop, mcdrop_mean_variance
# from LambdaZero.examples.drug_comb.new_drugcomb_data_v2 import DrugCombEdge

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()


data_config = {
    "target": "gridscore",
    "dataset_creator": LambdaZero.inputs.dataset_creator_v1,
    "dataset_split_path": osp.join(datasets_dir,
                                 "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),
    "dataset": LambdaZero.inputs.BrutalDock,
    "dataset_config": {
        "root": osp.join(datasets_dir, "brutal_dock/mpro_6lze"),
        "props": ["gridscore", "smi"],
        "transform": T.Compose([LambdaZero.utils.Complete()]),
        "file_names": ["Zinc15_2k"],
                     #["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
    },
    "b_size": 40,
    "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-43.042, 7.057])
}


regressor_config = {
    "config":{
        "data": dict(data_config, **{"dataset_creator":None}),
        "lambda": 1e-11,
        "T": 20,
        "lengthscale": 1e-2,
        "uncertainty_eval_freq": 60,
        "train_iterations": 59,
        "model": LambdaZero.models.MPNNetDrop,
        "model_config": {"drop_data":False, "drop_weights": False, "drop_last":True, "drop_prob":0.1},
        "optimizer": torch.optim.Adam,
        "optimizer_config": {
            "lr": 0.001
        },
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
    }
}

################# UCT experiment #################
uct000 = {}

# # control: randomly acquire data large and small dataset
random001 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "epsilon":1000,
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

random002 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "epsilon":1000,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

# greedy
greedy001 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "epsilon":0,
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

greedy002 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "epsilon":0,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

uct003_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":grid_search(list(10**np.linspace(start=-2,stop=2,num=5))),
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

uct003_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "kappa":grid_search(list(10**np.linspace(start=-2,stop=2,num=5))),
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance 
                    }
                }
            }}}

uct004_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":1e-2,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

uct004_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "kappa":1e-2,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance 
                    }
                }
            }}}

# uct 005, 006, 006 repeat uct 004 but with different kappas
uct005_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":1e-1,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

uct005_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "kappa":1e-1,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance 
                    }
                }
            }}}

uct006_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":1,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

uct006_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "kappa":1,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance 
                    }
                }
            }}}


uct007_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":10,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            }}}

uct007_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "kappa":10,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance 
                    }
                }
            }}}


################# Thompson Sampling experiment ################# 
ts003_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            },
            
            }}

ts003_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance
                    }
                }
            }}}


ts004_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }
                }
            },
            
            }}

ts004_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "dataset_split_path":
                    osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance
                    }
                }
            }}}


# fixme - this config structure is overcomlicated
def fp_feat(loader):
    fps = np.stack([d.fp for d in loader.dataset], axis=0)
    return fps
# bayesian regresison on fingerprints
uct008 = {
    "acquirer_config": {
        "config": {
            "data": {"dataset_config":{
                "transform":T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()])
                }},
            "regressor": BRR,
            "regressor_config": {"config": {
                "regressor_config":{"get_feat":fp_feat},
                "data":{"dataset_config":{
                    "transform":T.Compose([LambdaZero.utils.Complete(),LambdaZero.utils.MakeFP()])
                }}}}}}}

def concat_fp_feat(loader):
    feat = [np.concatenate([d.row_fp, d.col_fp]) for d in loader.dataset]
    return np.asarray(feat)

# comb_data = {"dataset": DrugCombEdge,
#                   "dataset_split_path": osp.join(datasets_dir, "NewDrugComb/raw/1700_split.npy"),
#                   "target": "negative_css",
#                   "dataset_config": {},
#                   "b_size": 40,
#                   "normalizer": LambdaZero.utils.MeanVarianceNormalizer([0, 1.]),
#                   }
# uctComb001 = {
#     "acquirer_config": {
#         #"stop": {"training_iteration": 115},
#         "config": {
#             "data": comb_data,
#             "regressor": BRR,
#             "regressor_config": {"config": {
#                 "data":comb_data,
#                 "regressor_config": {"get_feat": concat_fp_feat},
#             }},
#         "aq_size": 100,
#         }}}
# uctComb002 = uctComb001
# uctComb002["acquirer_config"]["config"]["epsilon"] = 10000