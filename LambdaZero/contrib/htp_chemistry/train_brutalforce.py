import sys
import os, os.path as osp
import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.tune.logger import DEFAULT_LOGGERS

import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.contrib.loggers import WandbRemoteLoggerCallback, RemoteLogger, TrialNameCreator
import train_brutalforce_config
from train_brutalforce_config import DEFAULT_CONFIG

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "brutalforce_config_002"
config = getattr(train_brutalforce_config, config_name)
config = merge_dicts(DEFAULT_CONFIG, config)

# # convenience option to debug option to be able to run any config on someone's machine
# machine = socket.gethostname()
# if machine == "Ikarus":
#     config = merge_dicts(DEFAULT_CONFIG, config_rlbo.debug_config)

if __name__ == "__main__":
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])

    # initialize loggers
    os.environ['WANDB_DIR'] = summaries_dir
    os.environ["WANDB_MODE"] = "dryrun"
    remote_logger = RemoteLogger.remote()
    #time.sleep(10) # this might not be needed, but added due to unsolved wandb init errors
    wandb_logger = WandbRemoteLoggerCallback(
        remote_logger=remote_logger,
        project=config["tune_config"]["config"]["logger_config"]["wandb"]["project"],
        api_key_file=osp.join(summaries_dir, "wandb_key"))
    if config_name == 'brutalforce_config_001':
        # not RLLib agent, just a tune.Trainable
        config["tune_config"]['config']["reward_config"]["scoreProxy_config"]["logger"] = remote_logger
        config["tune_config"]['config']["reward_config"]["scoreProxy_config"]["acquirer_config"]\
            ["model_config"]["logger"] = remote_logger
        config["tune_config"]["loggers"] = DEFAULT_LOGGERS + (wandb_logger,)

        # initialize scoreProxy which would be shared across many agents
        scoreProxy = config["tune_config"]['config']['reward_config']['scoreProxy'].\
            options(**config["tune_config"]['config']['reward_config']['scoreProxy_options']).\
            remote(**config["tune_config"]['config']['reward_config']['scoreProxy_config'])

        config["tune_config"]['config']['reward_config']['scoreProxy'] = scoreProxy
    else:
        config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"][
            "logger"] = remote_logger
        config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["acquirer_config"] \
            ["model_config"]["logger"] = remote_logger
        config["tune_config"]["loggers"] = DEFAULT_LOGGERS + (wandb_logger,)

        # initialize scoreProxy which would be shared across many agents
        scoreProxy = config["tune_config"]['config']['env_config']['reward_config']['scoreProxy']. \
            options(**config["tune_config"]['config']['env_config']['reward_config']['scoreProxy_options']). \
            remote(**config["tune_config"]['config']['env_config']['reward_config']['scoreProxy_config'])
        config["tune_config"]['config']['env_config']['reward_config']['scoreProxy'] = scoreProxy
    tune.run(**config["tune_config"], trial_name_creator=TrialNameCreator(config_name))