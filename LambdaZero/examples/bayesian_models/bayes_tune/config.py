import numpy as np
from ray.tune import grid_search
from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mcdrop, train_mpnn_brr, \
    mpnn_brr_mean_variance

mcdrop000 = {}

mcdrop001 = {"regressor_config":
                 {"config":
                      {"lambda": grid_search(list(0.1**np.linspace(start=7,stop=9,num=10)))}
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
#  python example_mcdrop.py mcdrop003

# (2) mpnn + drop; adjust lambda on small + large datasets
# python example_mcdrop.py mcdrop004
mcdrop004= {"regressor_config":
                 {"config":
                      {"lambda":grid_search(list(10 ** np.linspace(-1,-12,num=13)))}
                  },
             }
             
# (3) mpnn + make dropout in all layers a hyperparameter
# add dropout layer on input and all layers in torch_models.py. Hyperparameter drop_p already exists
