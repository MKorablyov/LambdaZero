import sys, time, socket
import os, os.path as osp
import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import merge_dicts
from ray.tune.logger import DEFAULT_LOGGERS

from guacamol.scoring_function import ArithmeticMeanScoringFunction

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.contrib.loggers import WandbRemoteLoggerCallback, RemoteLogger, TrialNameCreator
from LambdaZero.examples.baselines.guacamol_baseline.guacamol_baseline_config import DEFAULT_CONFIG
from LambdaZero.examples.baselines.guacamol_baseline import guacamol_baseline_config
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "lstm_001"
config = getattr(guacamol_baseline_config,config_name)
config = merge_dicts(DEFAULT_CONFIG, config)


if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)
    # initialize loggers
    os.environ['WANDB_DIR'] = summaries_dir
    os.environ["WANDB_MODE"] = "dryrun"
    remote_logger = RemoteLogger.remote()
    wandb_logger = WandbRemoteLoggerCallback(
        remote_logger=remote_logger,
        project=config["tune_config"]["config"]["logger_config"]["wandb"]["project"],
        api_key_file=osp.join(summaries_dir, "wandb_key"))
    config["tune_config"]['config']["reward_config"]["scoreProxy_config"][
        "logger"] = remote_logger
    config["tune_config"]['config']["reward_config"]["scoreProxy_config"]["oracle_config"] \
        ["logger"] = remote_logger
    config["tune_config"]['config']["reward_config"]["scoreProxy_config"]["acquirer_config"] \
        ["model_config"]["logger"] = remote_logger
    config["tune_config"]["loggers"] = DEFAULT_LOGGERS + (wandb_logger,)

    # initialize scoreProxy which would be shared across many agents
    scoreProxy = config["tune_config"]['config']['reward_config']['scoreProxy']. \
        options(**config["tune_config"]['config']['reward_config']['scoreProxy_options']). \
        remote(**config["tune_config"]['config']['reward_config']['scoreProxy_config'])

    config["tune_config"]['config']['reward_config']['scoreProxy'] = scoreProxy
    # run
    tune.run(**config["tune_config"], trial_name_creator=TrialNameCreator(config_name))


#
# if __name__ == "__main__":
#     ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
#     ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)
#     # initialize loggers
#     os.environ['WANDB_DIR'] = summaries_dir
#     os.environ["WANDB_MODE"] = "dryrun"
#     remote_logger = RemoteLogger.remote()
#     # this might not be needed, but added due to unsolved wandb init errors
#     wandb_logger = WandbRemoteLoggerCallback(
#         remote_logger=remote_logger,
#         project=config["logger_config"]["wandb"]["project"],
#         api_key_file=osp.join(summaries_dir, "wandb_key"))
#     config["reward_config"]["scoreProxy_config"][
#         "logger"] = remote_logger
#     config["reward_config"]["scoreProxy_config"]["oracle_config"] \
#         ["logger"] = remote_logger
#     config["reward_config"]["scoreProxy_config"]["acquirer_config"] \
#         ["model_config"]["logger"] = remote_logger
#     config["loggers"] = DEFAULT_LOGGERS + (wandb_logger,)
#
#     optimizer = config["method"](**config["method_config"])
#     scoring_fn = config["evaluator"](config) # config["evaluator"](config["reward"], config["reward_config"])
#     scoring_fn = ArithmeticMeanScoringFunction([scoring_fn])
#     optimized_molecule = optimizer.generate_optimized_molecules(scoring_fn, number_molecules=config["number_molecules"], starting_population=[])
#
#     print(optimized_molecule[0:32])
