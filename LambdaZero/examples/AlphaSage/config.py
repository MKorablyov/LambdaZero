from ray.tune import grid_search

pg001 = {
    "regressor_config": {
        "config": {
            "eps": grid_search([0.0, 0.01, 0.03, 0.09, 0.27, 0.9, 1000]),
            "t": grid_search([1,3,5,7,9,11,13]),
        }}}