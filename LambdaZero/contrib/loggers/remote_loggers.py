import ray
import numpy as np
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple
from ray.tune.utils import flatten_dict
from ray.tune.integration.wandb import WandbLoggerCallback, _clean_log, _WandbLoggingProcess, _WANDB_QUEUE_END
import wandb

@ray.remote
class RemoteLogger:
    def __init__(self):
        self.logs = []

    def log_wandb_object(self, name, data, object_type):
        # todo: wandb object is not recognized/logged possibly due to separate wandb.init call below is workaround
        if object_type == "graph":
            data = wandb.Graph(data)
        elif object_type == "histogram":
            data = wandb.Histogram(data)
        elif object_type == "image":
            data = wandb.Image(data)
        elif object_type == "molecule":
            data = wandb.Molecule(data)
        elif object_type == "plotly":
            data = wandb.Plotly(data)
        elif object_type == "object3D":
            data = wandb.Object3D(data)
        elif object_type == "table":
            data = wandb.Table(dataframe=data)
        else:
            raise NotImplementedError("not supported data type")
        self.logs.append({name: data})

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


class _WandbLoggingProcess_v2(_WandbLoggingProcess):
    def run(self):
        wandb.init(*self.args, **self.kwargs)
        while True:
            result = self.queue.get()
            if result == _WANDB_QUEUE_END:
                break
            elif type(result) == list: # remote logger logs lists that go to wandb directly
                for log in result:
                    wandb.log(log)
            else:
                log, config_update = self._handle_result(result)
                wandb.config.update(config_update, allow_val_change=True)
                wandb.log(log)
        wandb.join()

class WandbRemoteLoggerCallback(WandbLoggerCallback):
    _logger_process_cls = _WandbLoggingProcess_v2
    def __init__(self, remote_logger, **kwargs):
        self.remote_logger = remote_logger
        WandbLoggerCallback.__init__(self, **kwargs)

    def log_trial_result(self, iteration, trial, result):
        # log results to wandb the regular way
        WandbLoggerCallback.log_trial_result(self, iteration, trial, result)

        # log results from remote logger
        logs = ray.get(self.remote_logger.flush_logs.remote())
        print("obtained logs from the remote logger", logs)

        #for log in logs:
        self._trial_queues[trial].put(logs)


