import sys, time, socket
import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils import merge_dicts

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.examples.bayesian_models.rl import config

from LambdaZero.contrib.config_rlbo import rllib_config

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "ppo_bayes_reward_008"
config = getattr(config,config_name)
curr_trial = config_name + time.strftime("%Y-%m-%d_%H-%M-%S")


DEFAULT_CONFIG = {
    "rllib_config":rllib_config,
    "summaries_dir": summaries_dir,
    "memory": 30*10**9,
    "object_store_memory": 30*10**9,
    "trainer": PPOTrainer,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 2000000},
}

config = merge_dicts(DEFAULT_CONFIG, config)

# convenience option to debug on someone's laptop (add yours)
machine = socket.gethostname()
if machine == "Ikarus":
    config["rllib_config"]["num_workers"] = 2
    config["rllib_config"]["num_gpus"] = 0.3
    config["rllib_config"]["num_gpus_per_worker"] = 0.1
    config["memory"] = 10 * 10**9
    config["object_store_memory"] = 10 * 10**9
    config["rllib_config"]["train_batch_size"] = 10
    config["rllib_config"]["sgd_minibatch_size"] = 4



if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)

    # initialize scoreProxy which would be shared across many agents
    scoreProxy = config['rllib_config']['env_config']['reward_config']['scoreProxy'].remote(
        **config['rllib_config']['env_config']['reward_config']['scoreProxy_config'])
    config['rllib_config']['env_config']['reward_config']['scoreProxy'] = scoreProxy

    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])