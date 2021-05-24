from torch.optim import Adam, SGD, LBFGS

adam_v0 = {
    "type": Adam,
    "config": {
        "lr": 1e-4
    }
}

adam_v1 = {
    "type": Adam,
    "config": {
        "lr": 1e-3
    }
}
