import numpy as np
from ray.tune import grid_search
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mcdrop, train_mcdrop_brr

mcdrop000 = {}

mcdrop001 = {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(0.1**np.linspace(start=7,stop=9,num=10)))}
                  },
             }

mcdrop002 = {"regressor_config":
                 {"config":
                      {"T":50}
                  },
             }

mcdrop003 = {"regressor_config":
                 {"config":
                      {"train":train_mcdrop_brr}
                  },
             }


# "regressor_config":{
#         "run_or_experiment": MCDrop,
#         "config": {
#             "target": "gridscore",
#             "dataset_creator": _dataset_creator,
#             "dataset_split_path": osp.join(datasets_dir,
#                                            "brutal_dock/mpro_6lze/raw/randsplit_Zinc15_2k.npy"),