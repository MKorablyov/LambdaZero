import os

import ray
from ray import tune

from LambdaZero.examples.chemprop.ChempropRegressor import ChempropRegressor
from LambdaZero.trainable.early_stopping_stopper import EarlyStopping
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

model_parameters = {
    "name": "chemprop",
    "bias": False,
    "hidden_size": 300,
    "depth": 3,
    "dropout": 0.0,
    "atom_messages": False,
    "undirected": False,
    "ffn_hidden_size": 300,
    "ffn_num_layers": 2,
}

config = {
    "experiment_name": "TuneChemprop",
    "dataset_root": os.path.join(datasets_dir, "brutal_dock/d4"),
    "random_seed": 0,
    "target": "gridscore",
    "target_norm": [-26.3, 12.3],
    "lr": 0.001,
    "batch_size": 64,
    "model_parameters": model_parameters,
}


if __name__ == "__main__":
    ray.init()

    stopper = EarlyStopping(
        metric="validation_rmse_original_units",
        std=0.05,
        top=10,
        mode="min",
        patience=5,
    )

    analysis = tune.run(
        ChempropRegressor,
        config=config,
        stop=stopper,
        resources_per_trial={"cpu": 4, "gpu": 0.0},
        num_samples=1,
        checkpoint_at_end=False,
        local_dir=summaries_dir,
        checkpoint_freq=0,
    )
