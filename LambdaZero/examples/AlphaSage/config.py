from ray.tune import grid_search


pg000 = {
    "regressor_config": {
        "config": {
            "eps":1000,                   # noise
            "t": 3,                       # number of diffusion steps
            "optimizer_config": {"lr": 0.001,},
        }}}


pg001 = {
    "regressor_config": {
        "config": {
            "eps": grid_search([0.03, 0.09, 0.27, 1000]),                   # noise
            "t": grid_search([0,1,5,7,9]),                                  # number of diffusion steps
            "optimizer_config": {"lr": grid_search([0.001, 0.0001]),},
        }}}



pg002 = {
    "regressor_config": {
        "config": {
            "eps": grid_search([0.05, 1000]),
            "t": grid_search([5]),
            "optimizer_config": {"lr": grid_search([0.001, 0.0001]),},
        }}}


pg003 = {
    "regressor_config": {
        "config": {
            "eps": 0.03,
            "t": 2,
            "optimizer_config": {"lr": 0.001},
        }}}