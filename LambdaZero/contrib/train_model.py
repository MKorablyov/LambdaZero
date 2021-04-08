import time
import sys, os, os.path as osp
from copy import deepcopy
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.rllib.utils import merge_dicts
import LambdaZero.utils
from LambdaZero.contrib.loggers import RemoteLogger, WandbRemoteLoggerCallback, TrialNameCreator
from LambdaZero.contrib import config_model
from LambdaZero.contrib.config_model import DEFAULT_CONFIG

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


if __name__ == "__main__":
    # create config to run
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    else:
        config_name = "model_001"
    config = getattr(config_model, config_name)
    config = merge_dicts(DEFAULT_CONFIG, config)

    # initialize loggers
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    os.environ['WANDB_DIR'] = summaries_dir
    os.environ["WANDB_MODE"] = "dryrun"
    remote_logger = RemoteLogger.remote()
    # wandb_logger = WandbRemoteLoggerCallback(
    #     remote_logger=remote_logger,
    #     project=config["tune_config"]["config"]["logger_config"]["wandb"]["project"],
    #     api_key_file=osp.join(summaries_dir, "wandb_key"))

    config["tune_config"]['config']["model_config"]["logger"] = remote_logger
    # config["tune_config"]["loggers"] = DEFAULT_LOGGERS + (wandb_logger,)
    tune.run(**config["tune_config"], trial_name_creator=TrialNameCreator(config_name))