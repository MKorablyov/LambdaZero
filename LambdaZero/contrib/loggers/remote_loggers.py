import ray
from copy import deepcopy
from ray.tune.integration.wandb import WandbLoggerCallback, _clean_log
import os

@ray.remote
class RemoteLogger:
    def __init__(self):
        self.logs = []

    def log(self, logs):
        print("logger logs", logs)
        if type(logs) == list:
            self.logs.extend(logs)
        elif type(logs) == dict:
            self.logs.extend([logs])
        else:
            raise TypeError("only only logs list or dictionary")

    def flush_logs(self):
        logs = deepcopy(self.logs)
        self.logs = []
        return logs

    def config_name(self):
        return self._config_name

    def set_config(self, cfgn):
        self._config_name = cfgn

    def log_dir(self):
        return self._log_dir

    def set_log_dir(self, log_dir):
        self._log_dir = log_dir


class WandbRemoteLoggerCallback(WandbLoggerCallback):
    def __init__(self, remote_logger, **kwargs):
        self.remote_logger = remote_logger
        WandbLoggerCallback.__init__(self, **kwargs)

    def log_trial_result(self, iteration, trial, result):
        # log results to wandb the regular way
        WandbLoggerCallback.log_trial_result(self,iteration, trial, result)
        # log results from remote logger
        logs = ray.get(self.remote_logger.flush_logs.remote())
        print("obtained logs from the remote logger", logs)
        for log in logs:
            log = _clean_log(log)
            self._trial_queues[trial].put(log)


