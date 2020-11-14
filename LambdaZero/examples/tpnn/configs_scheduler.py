from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR


plateau_v0 = {
    "type": ReduceLROnPlateau,
    "config": {
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6
    }
}

cyclic_v0 = {
    "type": CyclicLR,
    "config": {
        "base_lr": 1e-6,
        "max_lr": 1e-3
    }
}
