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
            "epsilon":1000
        }}}