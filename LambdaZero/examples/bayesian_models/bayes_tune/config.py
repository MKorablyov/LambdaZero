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

mcdrop000 = {}

# mcdrop001 = {"regressor_config":
#                  {"config":
#                       {"lambda":grid_search(list(0.1**np.linspace(start=7,stop=9,num=10))),
#                       "model_config": {"do_dropout": True, "dropout_in_data": True, "drop_etc": True},
#                       "dataset_split_path": small_path,
#                        "dataset_config": {
#                          "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
#                          "props": ["gridscore", "smi"],
#                          "file_names": small_file_name}
#                       }
#                   },
#              }

# mcdrop002 = {"regressor_config":
#                  {"config":
#                       {"T":50,
#                       "model_config": {"do_dropout": True, "dropout_in_data": True, "drop_etc": True},
#                       "dataset_split_path": small_path,
#                        "dataset_config": {
#                          "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
#                          "props": ["gridscore", "smi"],
#                          "file_names": small_file_name}
#                       }
#                   },
#              }

# todo:
# (1) mpnn_brr large & small datasets 
# small: python example_mcdrop.py mcdrop003
# large: python example_mcdrop.py mcdrop004
mcdrop003 = {"regressor_config":
                 {"config":
                      {"train":train_mpnn_brr,
                       "get_mean_variance": mpnn_brr_mean_variance,
                       "model_config": {"do_dropout" : True, "dropout_in_data" : True, "drop_etc" : True},
                       "dataset_split_path": small_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": small_file_name}
                       
                       }
                  },
             }

mcdrop004 = {"regressor_config":
                 {"config":
                      {"train":train_mpnn_brr,
                       "get_mean_variance": mpnn_brr_mean_variance,
                       "model_config": {"do_dropout" : True, "dropout_in_data" : True, "drop_etc" : True},
                       "dataset_split_path": big_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": big_file_name}
                       
                       }
                  },
             }

# (2) mpnn + drop; adjust lambda on small + large datasets
# small: python example_mcdrop.py mcdrop005
# large: python example_mcdrop.py mcdrop006
mcdrop005= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-1,-12,num=13))),
                       "model_config": {"do_dropout" : True, "dropout_in_data" : True, "drop_etc" : True},
                       "dataset_split_path": small_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": small_file_name}
                      }
                  },
             }

mcdrop006= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-1,-12,num=13))),
                       "model_config": {"do_dropout" : True, "dropout_in_data" : True, "drop_etc" : True},
                       "dataset_split_path": big_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": big_file_name}
                      }
                  },
             }
             
# (3) mpnn + make dropout in all layers a hyperparameter
# add dropout layer on input and all layers in torch_models.py

#  on small dataset
#  dropout all on: python example_mcdrop.py mcdrop007
#  dropout only on weights: python example_mcdrop.py mcdrop008
#  dropout only on data: python example_mcdrop.py mcdrop009
#  dropout only on last layer: python example_mcdrop.py mcdrop010
mcdrop007= {"regressor_config":
                 {"config":
                      {"model_config": {"do_dropout" : True, "dropout_in_data" : True, "drop_etc" : True},
                       "dataset_split_path": small_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": small_file_name}
                      }
                  },
             }
 
mcdrop008= {"regressor_config":
                 {"config":
                      {"model_config": {"do_dropout" : False, "dropout_in_data" : False, "drop_etc" : True},
                       "dataset_split_path": small_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": small_file_name}
                      }
                  },
             }

mcdrop009= {"regressor_config":
                 {"config":
                      {"model_config": {"do_dropout" : False, "dropout_in_data" : True, "drop_etc" : False},
                       "dataset_split_path": small_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": small_file_name}
                      }
                  },
             }

mcdrop010= {"regressor_config":
                 {"config":
                      {"model_config": {"do_dropout" : True, "dropout_in_data" : False, "drop_etc" : False},
                       "dataset_split_path": small_path,
                       "dataset_config": {
                         "root": os.path.join(datasets_dir, "brutal_dock/mpro_6lze"),
                         "props": ["gridscore", "smi"],
                         "file_names": small_file_name}
                      }
                  },
             }
