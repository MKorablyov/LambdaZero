from LambdaZero.examples.bayesian_models.bayes_tune.functions import train_mpnn_brr, mpnn_brr_mean_variance

uct000 = {}

# this will apply mpnn_blr uncertainty model instead of mcdrop
uct001 = {
    "acquirer_config":{
        "config":{
            "regressor_config":{
                "config":
                    {"train": train_mpnn_brr,
                     "get_mean_variance": mpnn_brr_mean_variance}
            }}}}

# this will make acquision function random
uct002 = {
    "acquirer_config":{
        "config":{
            "epsilon": grid_search([0, 1000, np.linspace(start=0.01,stop=1,num=50)])
        }}}

# todo
# (1) mpnn bll + UCT + find kappa on small and large datasets
# python example_UCT uct003
uct003 = {"acquirer_config":
                 {"config":
                      {"train":train_mpnn_brr,
                       "get_mean_variance": mpnn_brr_mean_variance,
                         "kappa":grid_search(list(np.linspace(start=0.01,stop=100,num=20)))
                       }
                  },
             }

# (2) with the best kappa try different amounts of noise
# baseline without noise, random (epislon = 1000), greedy/egreedy (epislon between 0 and 1)
# python uct002

# (3) find the best alpha for mpnn + mcdrop
# see config.py

