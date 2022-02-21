from argparse import Namespace
from typing import Callable, Any

from LambdaZero.examples.gflow.datasets import data_generator
from LambdaZero.examples.gflow.datasets import data_generator_multiproc
from LambdaZero.examples.gflow.datasets import sample_1transition
from LambdaZero.examples.gflow.datasets import priority_sampling
from LambdaZero.examples.gflow.datasets import sample_offline_traj
from LambdaZero.examples.gflow.datasets import debug_flow
from LambdaZero.examples.gflow.datasets import data_with_test_set

TRAIN_DATASET = {
    "DataGenerator": data_generator.DataGenerator,
    "DataGeneratorMultiProc": data_generator_multiproc.DataGeneratorMultiProc,
    "DataGenSampleParentsTraj": sample_1transition.DataGenSampleParentsTraj,
    "PrioritySamplingData": priority_sampling.PrioritySamplingData,
    "BatchWithOfflineTraj": sample_offline_traj.BatchWithOfflineTraj,
    "DebugFlow": debug_flow.DebugFlow,
    "DataWithTestSet": data_with_test_set.DataWithTestSet,
}


def get_gflow_dataset(cfg: Namespace) -> Callable[[Any], data_generator.DataGenerator]:
    assert hasattr(cfg, "name") and cfg.name in TRAIN_DATASET,\
        f"Please provide a valid GFLOW Train Dataset. Error with {cfg.name}"
    return TRAIN_DATASET[cfg.name]
