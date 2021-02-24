import sys, time, socket
import os, os.path as osp
import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.utils import merge_dicts

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.contrib.loggers import WandbRemoteLoggerCalback, RemoteLogger
from LambdaZero.contrib.config_rlbo import DEFAULT_CONFIG
import config_rlbo
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "debug_config"
config = getattr(config_rlbo,config_name)
config = merge_dicts(DEFAULT_CONFIG, config)


# convenience option to debug option to be able to run any config on someone's machine
machine = socket.gethostname()
if machine == "Ikarus":
    config = merge_dicts(DEFAULT_CONFIG, config_rlbo.debug_config)



if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)

    # initialize loggers
    os.environ['WANDB_DIR'] = summaries_dir
    os.environ["WANDB_MODE"] = "dryrun"
    remote_logger = RemoteLogger.remote()
    wandb_callback = WandbRemoteLoggerCalback(
        remote_logger=remote_logger,
        project="Optimization_Project",
        api_key_file=osp.join(summaries_dir,"wandb_key"),
        log_config=False)
    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["acquirer_config"]\
        ["model_config"]["logger"] = remote_logger

    # initialize scoreProxy which would be shared across many agents
    scoreProxy = config["tune_config"]['config']['env_config']['reward_config']['scoreProxy'].remote(
        **config["tune_config"]['config']['env_config']['reward_config']['scoreProxy_config'])
    config["tune_config"]['config']['env_config']['reward_config']['scoreProxy'] = scoreProxy

    tune.run(**config["tune_config"])