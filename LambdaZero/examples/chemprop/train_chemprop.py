import os

import ray
from ray import tune

import LambdaZero.utils
from LambdaZero.examples.chemprop.ChempropRegressor import ChempropRegressor
from LambdaZero.examples.chemprop.epoch_trainers import train_epoch, eval_epoch
from LambdaZero.utils import get_external_dirs

transform = LambdaZero.utils.Complete()
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

DEFAULT_CONFIG = {
    "trainer": ChempropRegressor,
    "trainer_config": {
        "dataset_root": os.path.join(datasets_dir, "brutal_dock/d4"),
        "targets": ["gridscore"],
        "target_norms": [[-26.3, 12.3]],
        "file_names": ["dock_blocks105_walk40_clust"],
        "transform": transform,
        "split_name": "randsplit_dock_blocks105_walk40_clust",
        "lr": 0.001,
        "b_size": 64,
        "dim": 64,
        "num_epochs": 120,

        #"model": "some_model", todo

        "molprops": ["gridscore", "klabel"],
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        # todo: test epoch
        },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 250000000,
    "stop": {"training_iteration": 2},
}

config = DEFAULT_CONFIG


if __name__ == "__main__":
    ray.init()

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop={"training_iteration": 10}, #EarlyStop(),
                        resources_per_trial={
                           "cpu": 1, # fixme requesting all CPUs blocks additional call to ray from LambdaZero.input
                           "gpu": 0.0
                        },
                        num_samples=1,
                        checkpoint_at_end=False,
                        local_dir=summaries_dir,
                        checkpoint_freq=100000)
