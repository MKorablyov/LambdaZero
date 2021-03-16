import torch

qm9_1k = {
    "target": 1,
    "dry_run": False,
    "control_exp": False,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "train_dataset_size": False,
    "infer_edges": False,
    "settoset": False,
    "kNN": False,
    "training_iterations": 500,
}

qm9_all = {
    "target": 1,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": False,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": False,
    "kNN": False,
    "training_iterations": 1000,
}

qm9_all_poscontrol = {
    "target": 1,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": True, # randomly assigns positions
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": False,
    "kNN": False,
    "training_iterations": 1000,
}

qm9_all_noscheduler = {
    "target": 1,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": False,
    "scheduler": None,
    "infer_edges": False,
    "settoset": False,
    "kNN": False,
    "training_iterations": 1000,
}

qm9_set2set = {
    "target": 1,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": False,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": True,
    "kNN": False,
    "training_iterations": 1000,
}

qm9_distasedges = { # need to redo with proper infer edges
    "target": 1,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": False,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": False,
    "kNN": True,
    "training_iterations": 1000,
}

qm9_distasedge_poscontrol = {
    "target": 1,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": True,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": False,
    "kNN": True,
    "training_iterations": 1000,
}

qm9_noinferedge = {
    "target": 1,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": False,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": False,
    "kNN": False,
    "training_iterations": 1000,
}

qm9_10k = {
    "target": 1,
    "train_dataset_size": 0.10,
    "dry_run": True,
    "control_exp": False,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": False,
    "kNN": False,
    "training_iterations": 1000,
}

qm9_7_all = {
    "target": 7,
    "train_dataset_size": 0.75,
    "dry_run": True,
    "control_exp": False,
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "infer_edges": False,
    "settoset": False,
    "kNN": False,
    "training_iterations": 1000,
}