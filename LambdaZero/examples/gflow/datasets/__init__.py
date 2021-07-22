import cv2
from argparse import Namespace
from typing import Callable, Any

from LambdaZero.examples.gflow.datasets import data_generator
from LambdaZero.examples.gflow.datasets import mprocs_datafeed
from LambdaZero.examples.gflow.datasets import sample_1transition

TRAIN_DATASET = {
    "DataGenerator": data_generator.DataGenerator,
    "OnlineDataFeed": mprocs_datafeed.OnlineDataFeed,
    "OnlineDataFeedTransition": sample_1transition.OnlineDataFeedTransition
}


def get_gflow_dataset(cfg: Namespace) -> Callable[[Any], data_generator.DataGenerator]:
    assert hasattr(cfg, "name") and cfg.name in TRAIN_DATASET,\
        "Please provide a valid GFLOW Train Dataset."
    return TRAIN_DATASET[cfg.name]
