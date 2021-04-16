import sys, time, socket
import os, os.path as osp
import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.tune.logger import DEFAULT_LOGGERS

import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.contrib.loggers import WandbRemoteLoggerCallback, RemoteLogger, TrialNameCreator
from LambdaZero.contrib.config_random_policy import DEFAULT_CONFIG
import config_random_policy
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "rand_001"
config = getattr(config_random_policy, config_name)
config = merge_dicts(DEFAULT_CONFIG, config)

# also make it work on one GPU and less RAM when on Maksym's machine
machine = socket.gethostname()
if machine == "Ikarus":
    config = merge_dicts(config, config_random_policy.debug_config)


if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    # initialize loggers
    os.environ['WANDB_DIR'] = summaries_dir
    os.environ["WANDB_MODE"] = "dryrun"
    remote_logger = RemoteLogger.remote()
    wandb_logger = WandbRemoteLoggerCallback(
        remote_logger=remote_logger,
        project=config["tune_config"]["config"]["logger_config"]["wandb"]["project"],
        api_key_file=osp.join(summaries_dir, "wandb_key"))
    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"][
        "logger"] = remote_logger
    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["oracle_config"] \
        ["logger"] = remote_logger
    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["acquirer_config"] \
        ["model_config"]["logger"] = remote_logger
    config["tune_config"]["loggers"] = DEFAULT_LOGGERS + (wandb_logger,)

    # initialize scoreProxy which would be shared across many agents
    scoreProxy = config["tune_config"]['config']['env_config']['reward_config']['scoreProxy']. \
        options(**config["tune_config"]['config']['env_config']['reward_config']['scoreProxy_options']). \
        remote(**config["tune_config"]['config']['env_config']['reward_config']['scoreProxy_config'])

    config["tune_config"]['config']['env_config']['reward_config']['scoreProxy'] = scoreProxy
    # run
    tune.run(**config["tune_config"], trial_name_creator=TrialNameCreator(config_name))