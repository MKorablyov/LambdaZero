from argparse import Namespace
from typing import Callable, Any

from LambdaZero.examples.gflow.datasets import data_generator
from LambdaZero.examples.gflow.datasets import data_generator_multiproc
from LambdaZero.examples.gflow.datasets import sample_1transition
from LambdaZero.examples.gflow.datasets import priority_sampling
from LambdaZero.examples.gflow.datasets import sample_offline_traj
from LambdaZero.examples.gflow.datasets import data_no_online

TRAIN_DATASET = {
    "DataGenerator": data_generator.DataGenerator,
    "DataGeneratorMultiProc": data_generator_multiproc.DataGeneratorMultiProc,
    "DataGenSampleParentsTraj": sample_1transition.DataGenSampleParentsTraj,
    "PrioritySamplingData": priority_sampling.PrioritySamplingData,
    "BatchWithOfflineTraj": sample_offline_traj.BatchWithOfflineTraj,
    "DataFilterAddOnline": data_no_online.DataFilterAddOnline,
}


def get_gflow_dataset(cfg: Namespace) -> Callable[[Any], data_generator.DataGenerator]:
    assert hasattr(cfg, "name") and cfg.name in TRAIN_DATASET,\
        f"Please provide a valid GFLOW Train Dataset. Error with {cfg.name}"
    return TRAIN_DATASET[cfg.name]
