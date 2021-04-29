import sys, time, socket
import os, os.path as osp
import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import merge_dicts
from ray.tune.logger import DEFAULT_LOGGERS

from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.contrib.proxy import LogTrajectories
from LambdaZero.contrib.loggers import WandbRemoteLoggerCallback, RemoteLogger, TrialNameCreator
import config_rlbo

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    else:
        config_name = "rlbo4_014"
    config = getattr(config_rlbo, config_name)
    config = merge_dicts(config.pop("default"), config)

    # also make it work on one GPU and less RAM when on Maksym's machine
    machine = socket.gethostname()
    if machine == "Ikarus":
        config = merge_dicts(config, config_rlbo.config_debug)

    #for i in range(7):
    #    try:
            # Maksym Feb 26
            # I have been trying to debug the issue of ray sometimes resulting in GPU-OOM
            # what seems to happen is that ray jobs sometimes fail very soon after the initialization
            # same exact jobs can run for a while when initialized again. I think the issue is related to how individual
            # remote workers are allocated. Yet, I have not been able to entirely debug it. Therefore this for loop here
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)
    # initialize loggers
    os.environ['WANDB_DIR'] = summaries_dir
#    os.environ["WANDB_MODE"] = "dryrun"


    remote_logger = RemoteLogger.remote()



    wandb_logger = WandbRemoteLoggerCallback(
        remote_logger=remote_logger,
        project=config["tune_config"]["config"]["logger_config"]["wandb"]["project"],
        api_key_file=osp.join(summaries_dir, "wandb_key"))
    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"][
                "logger"] = remote_logger
    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["oracle_config"]\
                ["logger"] = remote_logger
    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["acquisition_config"] \
                ["model_config"]["logger"] = remote_logger

    config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["after_acquire"] = \
        config["tune_config"]['config']['env_config']["reward_config"]["scoreProxy_config"]["after_acquire"] + \
        [LogTrajectories(10, remote_logger)] # fixme -- not 10

    config["tune_config"]["loggers"] = DEFAULT_LOGGERS + (wandb_logger,)
    # initialize scoreProxy which would be shared across many agents
    scoreProxy = config["tune_config"]['config']['env_config']['reward_config']['scoreProxy'].\
        options(**config["tune_config"]['config']['env_config']['reward_config']['scoreProxy_options']).\
        remote(**config["tune_config"]['config']['env_config']['reward_config']['scoreProxy_config'])
    config["tune_config"]['config']['env_config']['reward_config']['scoreProxy'] = scoreProxy



    # run
    tune.run(**config["tune_config"], trial_name_creator=TrialNameCreator(config_name))
    # except Exception as e:
    #    print(e)
    ray.shutdown()