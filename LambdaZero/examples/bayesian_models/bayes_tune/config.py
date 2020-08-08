import numpy as np
from ray.tune import grid_search
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mcdrop, train_mpnn_brr, \
    mpnn_brr_mean_variance

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
                      {"train":train_mpnn_brr,
                       "get_mean_variance": mpnn_brr_mean_variance,

                       }
                  },
             }

# todo:
# (1) mpnn_brr large & small datasets
# (2) mpnn + drop; adjust alpha on small + large datasets
# (3) mpnn + make dropout in all layers a hyperparameter
