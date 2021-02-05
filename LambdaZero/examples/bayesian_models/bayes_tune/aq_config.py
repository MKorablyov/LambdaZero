import os.path as osp
import os
import numpy as np
import torch
from ray.tune import grid_search
from torch_geometric import transforms as T
import LambdaZero.utils
import LambdaZero.chem
import LambdaZero.inputs
import LambdaZero.models
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mcdrop, train_mpnn_brr, \
    mpnn_brr_mean_variance, train_deup_debug
from LambdaZero.examples.bayesian_models.bayes_tune.brr import BRR
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_epoch,eval_epoch, train_mcdrop, \
    mcdrop_mean_variance, train_mpnn_deup, deup_mean_variance
from LambdaZero.examples.drug_comb.new_drugcomb_data_v2 import DrugCombEdge

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()


data_config = {
    "target": "dockscore",
    "dataset_creator": LambdaZero.inputs.dataset_creator_v1,
    "dataset_split_path": osp.join(datasets_dir,
                                 "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k.npy"),
    "dataset": LambdaZero.inputs.BrutalDock,
    "dataset_config": {
        "root": osp.join(datasets_dir, "brutal_dock/seh"),
        "props": ["dockscore", "smiles"],
        "transform": T.Compose([LambdaZero.utils.Complete()]),
        "file_names": ["Zinc20_docked_neg_randperm_30k"],
                     #["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"],
    },
    "b_size": 40,
    "normalizer": LambdaZero.utils.MeanVarianceNormalizer([-8.6, 1.1])
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
                "kappa":0,
                "epsilon":1000,
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }}}}}



random002 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa": 0,
                "epsilon": 1000,
                "dataset_split_path":
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }}}}}
# greedy
greedy001 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "epsilon": 0,
                "kappa": 0,
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }}}}}

greedy002 = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "epsilon":0,
                "kappa":0,
                "dataset_split_path":
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }}}}}

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
                    }}}}}

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
                    }}}}}

uct004_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":1e-2,
                "dataset_split_path":
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }}}}}

uct004_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "kappa":1e-2,
                "dataset_split_path":
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance 
                    }}}}}

# uct 005, 006, 006 repeat uct 004 but with different kappas
uct005_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":1e-1,
                "dataset_split_path":
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mpnn_brr,
                        "get_mean_variance":mpnn_brr_mean_variance
                    }}}}}

uct005_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "kappa":1e-1,
                "dataset_split_path":
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
                "dataset_config": {
                    "file_names": ["Zinc15_260k_0", "Zinc15_260k_1", "Zinc15_260k_2", "Zinc15_260k_3"]},
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance 
                    }}}}}

uct006_brr = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": mpnn_brr_mean_variance,
                "kappa":1,
                "dataset_split_path":
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
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
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
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
                    osp.join(datasets_dir,
                             "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_260k_after_fixing_1_broken_mol.npy"),
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



ucb008_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": grid_search(list(10**np.linspace(start=-1.5,stop=1,num=5))),
                "regressor_config":{
                    "config":{
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": True,
                                         "drop_prob": 0.1, "out_dim": 2},

                        "train": train_mpnn_deup,
                        "get_mean_variance":deup_mean_variance
                    }}}}}



ucb009_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": grid_search(list(10**np.linspace(start=-1.5,stop=1,num=5))),
                "regressor_config":{
                    "config":{
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": True,
                                         "drop_prob": 0.1, "out_dim": 2},

                        "train": train_mpnn_deup,
                        "get_mean_variance":deup_mean_variance
                    }}}}}


ucb010_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": 0.07,
                "regressor_config":{
                    "config":{
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": True,
                                         "drop_prob": 0.1, "out_dim": 2},

                        "train": train_mpnn_deup,
                        "get_mean_variance":deup_mean_variance
                    }}}}}


ucb011_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": 0.5,
                "regressor_config":{
                    "config":{
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": True,
                                         "drop_prob": 0.1, "out_dim": 2},

                        "train": train_mpnn_deup,
                        "get_mean_variance":deup_mean_variance
                    }}}}}

ucb012_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": 0.5,
                "regressor_config":{
                    "config":{
                        "model":LambdaZero.models.MPNNetDropSmall,
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": False,
                                         "drop_prob": 0.0, "out_dim": 2},

                        #"train_iterations": 1451,
                        "train": train_mpnn_deup,
                        "get_mean_variance":deup_mean_variance
                    }}}}}

ucb013_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": 0.0,
                "regressor_config":{
                    "config":{
                        "model":LambdaZero.models.MPNNetDrop,
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": False,
                                         "drop_prob": 0.0, "out_dim": 2},

                        "train_iterations": 61,
                        "train": train_deup_debug,
                        "get_mean_variance":deup_mean_variance
                    }}}}}


ucb014_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": grid_search(list(10**np.linspace(start=-1.5,stop=1,num=8))),
                "regressor_config":{
                    "config":{
                        "model":LambdaZero.models.MPNNetDrop,
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": False,
                                         "drop_prob": 0.0, "out_dim": 2},

                        "train_iterations": 61,
                        "train": train_mpnn_deup,
                        "get_mean_variance":deup_mean_variance
                    }}},
        "stop": {"training_iteration": 1}
    }}
# +-----------------+------------+-------+------------+--------+------------------+-------------------+-------------------+------------+
# | Trial name      | status     | loc   |      kappa |   iter |   total time (s) |   aq_top15_regret |   aq_top50_regret |   aq_top15 |
# |-----------------+------------+-------+------------+--------+------------------+-------------------+-------------------+------------|
# | UCB_3bb76_00000 | TERMINATED |       |  0.0316228 |      1 |          282.681 |          0.7      |          0.599999 |      -13   |
# | UCB_3bb76_00001 | TERMINATED |       |  0.0719686 |      1 |          274.256 |          0.8      |          0.7      |      -12.9 |
# | UCB_3bb76_00002 | TERMINATED |       |  0.163789  |      1 |          268.965 |          0.8      |          0.799999 |      -12.9 |
# | UCB_3bb76_00003 | TERMINATED |       |  0.372759  |      1 |          254.992 |          0.5      |          0.7      |      -13.2 |
# | UCB_3bb76_00004 | TERMINATED |       |  0.848343  |      1 |          281.678 |          0.8      |          0.7      |      -12.9 |
# | UCB_3bb76_00005 | TERMINATED |       |  1.9307    |      1 |          279.885 |          0.599999 |          0.7      |      -13.1 |
# | UCB_3bb76_00006 | TERMINATED |       |  4.39397   |      1 |          309.035 |          0.4      |          0.599999 |      -13.3 |
# | UCB_3bb76_00007 | TERMINATED |       | 10         |      1 |          295.208 |          0.7      |          0.7      |      -13   |
# +-----------------+------------+-------+------------+--------+------------------+-------------------+-------------------+------------+


ucb015_deup = {
    "acquirer_config":{
            "config":{
                "train":train_mpnn_brr,
                "get_mean_variance": deup_mean_variance,
                "kappa": grid_search(list(10**np.linspace(start=-1.5,stop=1,num=8))),
                "regressor_config":{
                    "config":{
                        "model":LambdaZero.models.MPNNetDrop,
                        "model_config": {"drop_data": False, "drop_weights": False, "drop_last": False,
                                         "drop_prob": 0.0, "out_dim": 2},

                        "train_iterations": 61,
                        "train": train_mpnn_deup,
                        "get_mean_variance":deup_mean_variance
                    }}},
        "stop": {"training_iteration": 10}
    }}


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
                    }}}}}

ts003_mcdrop = {
    "acquirer_config":{
            "config":{
                "train":train_mcdrop,
                "get_mean_variance": mcdrop_mean_variance,
                "regressor_config":{
                    "config":{
                        "train": train_mcdrop,
                        "get_mean_variance": mcdrop_mean_variance
                    }}}}}


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
            "data": {"dataset_config": {
                "transform": T.Compose([LambdaZero.utils.Complete(), LambdaZero.chem.MakeFP()])
                }},
            "regressor": BRR,
            "regressor_config": {"config": {
                "regressor_config": {"get_feat": fp_feat},
                "data":{"dataset_config": {
                    "transform": T.Compose([LambdaZero.utils.Complete(), LambdaZero.chem.MakeFP()])
                }}}}}}}


def concat_fp_feat(loader):
    feat = [np.concatenate([d.row_fp, d.col_fp]) for d in loader.dataset]
    return np.asarray(feat)

comb_data = {"dataset": DrugCombEdge,
                  "dataset_split_path": osp.join(datasets_dir, "NewDrugComb/raw/1700_split.npy"),
                  "target": "negative_css",
                  "dataset_config": {},
                  "b_size": 40,
                  "normalizer": LambdaZero.utils.MeanVarianceNormalizer([0, 1.]),
                  }
uctComb001 = {
    "acquirer_config": {
        #"stop": {"training_iteration": 115},
        "config": {
            "data": comb_data,
            "epsilon": 0.0,#111110.0,
            #"minimize_objective": True,
            "kappa": 0.00,
            "aq_size": 100,
            # grid_search(list(10**np.linspace(start=-2,stop=2,num=32))),
            "regressor": BRR,
            "regressor_config": {"config": {
                "data":comb_data,
                "regressor_config": {"get_feat": concat_fp_feat},
            }},

        },
        "resources_per_trial": {"cpu": 4, "gpu": 0.0}
    }}

uctComb002 = {
    "acquirer_config": {
        "config": {
            "data": comb_data,
            "kappa": grid_search(list(10**np.linspace(start=-2,stop=2,num=32))),
            "aq_size": 100,
            "regressor": BRR,
            "regressor_config": {"config": {
                "data":comb_data,
                "regressor_config": {"get_feat": concat_fp_feat},
            }},

        },
        "resources_per_trial": {"cpu": 1, "gpu": 0.0}
    }}