from torch.optim import Adam, SGD, LBFGS

adam_v0 = {
    "type": Adam,
    "config": {
        "lr": 1e-4
    }
}

sgd_v0 = {
    "type": SGD,
    "config": {
        "lr": 1e-4
    }
}

lbfgs_v0 = {
    "type": LBFGS,
    "config": {
        "lr": 1e-4,
        "history_size": 10,
        "max_iter": 4,
        "line_search_fn": 'strong_wolfe'
    }
}

