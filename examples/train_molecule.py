import sys, argparse,time, socket,logging, os.path as osp
import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from affinity_models.alpha_zero.models.custom_torch_models import MolActorCritic_thv1
from affinity_models.alpha_zero.models.custom_tf_models import MolActorCritic_tfv1
from affinity_models.alpha_zero.examples.config import get_config, alphazero_config
if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "ppo001" # "ppo001"

trainer, config, memory, summaries_dir, checkpoint_freq = get_config(config_name)
print("starting with config:", config_name)


if __name__ == "__main__":
    ray.init(memory=memory)
             #object_store_memory=6*10**9,
             #driver_object_store_memory=3*10**9)
    ModelCatalog.register_custom_model("MolActorCritic_thv1", MolActorCritic_thv1)
    ModelCatalog.register_custom_model("MolActorCritic_tfv1", MolActorCritic_tfv1)
    tune.run(trainer,
        stop={"training_iteration": 2000000},
        max_failures=0,
        config=config,
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=checkpoint_freq)