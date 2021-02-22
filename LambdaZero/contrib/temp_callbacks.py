# todo: I would need to log results from proxyServer & model which are asynchronous
# todo: I think we could just create a separate wandb logger which would have a trial name/index
# todo: I need to find trial ID/name and log these

import time
import random
import numpy as np
from ray import tune
from ray.tune.integration.wandb import wandb_mixin
import wandb
import ray

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback, _clean_log, _set_api_key


import gym
from gym.spaces import Discrete,Box

from typing import Dict, List
import json
import os
from ray.tune.logger import LoggerCallback


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        print(config)
        self.end_pos = config["corridor_length"]
        self.logger = config["logger"]
        # logger_config = {"id":config["trial_id"],
        #                  "name":config["trial_name"],
        #                  "group":"PPO",
        #                  "project":"Optimization_Project",
        #                  "resume":True, "reinit":True, "allow_val_change":False}
        #self.wandb = wandb.init(**logger_config)
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1, ), dtype=np.float32)
        self.steps = 0
        print("env initialized!!!!!")

    def reset(self):
        self.cur_pos = 0
        self.logger.log([{"pos":self.cur_pos + random.random()}])
        print("logger in env", self.logger)
        #print("calling log!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return [self.cur_pos]

    def step(self, action):
        self.steps +=1
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return [self.cur_pos], 1.0 if done else -0.1, done, {"logger":self.logger}

import copy
class WandbLogger_v2(WandbLoggerCallback):
    def __init__(self, other_instance, **kwargs):
        #self._proxy_logs = [{"a":1}]
        #self.length = 0
        WandbLoggerCallback.__init__(self,**kwargs)
        #self.other_instance = other_instance
        #print("logger initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #time.sleep(3)

    #def log(self, result):
        #print("log is called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
        #print(result, self.proxy_logs)
        #self._proxy_logs.extend(copy.deepcopy(result))
        #self.length +=1
        #print(self._proxy_logs)


    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):


        #print(ray.get(trial.runner.get_config.remote())["env_config"]["logger"].logs)
        #print(trial.runner.__dict__)
        #print(trial.__dict__.keys())
        #print(trial.config["env_config"]["logger"].logs)
        #print("trial result", result["config"]["env_config"]["logger"].logs)
        #for key, value in trial.runner.__dict__.items():
        #    print(key, ":", value)
        #print(ray.get(trial.runner.get_logger.remote()))
        #time.sleep(100)

        print("trial result copies", result["config"]["env_config"]["logger"].copies)
        #time.sleep(100)
        #print("proxy logs", self._proxy_logs, self.length)
        #proxy_logs = copy.deepcopy(self.proxy_logs)
        #self.proxy_logs = []
        #print("result", result)
        #print("proxy logs!!!!!!!!!!!!!!!!!!", self._proxy_logs )
        #print(self.other_instance.logs)

        #print()
        if trial not in self._trial_processes:
            self.log_trial_start(trial)
        result = _clean_log(result)
        self._trial_queues[trial].put(result)

        #for i in range(len(self._proxy_logs)):
        #    r = _clean_log(self._proxy_logs[i])
        #    self._trial_queues[trial].put(r)

    #def __del__(self):
        #print("this object being deleted")
        #time.sleep(10)

    # todo: Ideally I would still like to pass logger and say logger.log.remote()
    # but I would not know Trial

from copy import deepcopy
class OtherInstance:
    def __init__(self):
        self.copies = 0
        print("init oyhjert instace")
        self.logs = [{"other_instance":1}]
    def log(self, logs):
        self.logs.extend(logs)
        print(logs)
        #print(self.logs)

    #def __del__(self):
    #    print("trying to delete other instance")
        #time.sleep(10)

    #def __deepcopy__(self, memo):
    #    cls = self.__class__
    #    result = cls.__new__(cls)
    #    memo[id(self)] = result
    #    for k, v in self.__dict__.items():
    #        setattr(result, k, deepcopy(v, memo))
     #   result.copies +=1
    #    return result


other = OtherInstance()
wandb_logger = WandbLogger_v2(other_instance=other,
                              project="Optimization_Project",
                              api_key_file="/home/maksym/Summaries/wandb_key",
                              log_config=False)

from ray.tune import Callback
class MyCallback(Callback):
    def __init__(self):
        Callback.__init__(self)
        self.logs = []

    def on_trial_result(self, iteration: int, trials: List["Trial"], trial: "Trial", result: Dict, **info):
        #print("info", info)
        #time.sleep(100)
        #print("logger logs", trial.config["env_config"]["logger"].logs)
        print(trial.__dict__)

        #print(trial.__dict__)
        #print(trial.runner.__dict__)
        #time.sleep(100)

        #trials[0].config["env_config"]["trial_id"] = trials[0].trial_id
        #print("step brgin !!!!!!!!!!!!!!")
        #time.sleep(1000)
        print("trial", trial, self.logs)

   # def __del__(self):
   #     print("trying to delete my calback!!!!")
   #     time.sleep(3)

    #def log(self,log):
    #    self.logs.extend(log)


class PPOTrainerv2(PPOTrainer):
    #def __init__(self, **kwargs):
    #    PPOTrainer.__init__(self, "lolo")
        #raise Exception

    def _make_workers(self, **kwargs):
        print("logger when making workers", kwargs["config"]["env_config"]["logger"])
        workers = PPOTrainer._make_workers(self,**kwargs)
        #print(workers.remote_workers()[0].__dict__)
        #time.sleep(100)
        return workers

    #def get_logger(self):
        #print(self.env_creator().__dict__)
        #for k,v in self.workers._remote_workers().__dict__.items():
        #    print(k, "------", v)
        #print(self.workers._remote_workers[0].__dict__)
        #time.sleep(100)

        #print("ppo trainer v2 dict", self.__dict__)
        #print(self.env.logger)
        #for key,value in self.__dict__.items():
        #   print(key,":---:", value)
        #print(self.config["env_config"]["logger"].logs)
        #time.sleep(100)
        #return logger


        #print("======= env creator ================")
        #print(self.env_creator.__dict__)
        #print(self.raw_user_config["env_config"]["logger"].logs)

        #time.sleep(1000)

mycallback = MyCallback()
logger = OtherInstance()
tune.run(
    PPOTrainerv2,
    checkpoint_freq=1,
    config={
        "framework": "torch",
        "num_workers": 2,
        "num_gpus": 1,
        "env": SimpleCorridor,
        "env_config":{"corridor_length":7, "logger":logger},

        },
    stop={"training_iteration":3},
    loggers=DEFAULT_LOGGERS + (wandb_logger,),
    callbacks=[mycallback]
)










# wandb_config = {'id': '6e51e_00005',
#                 'name': 'PPO_SimpleCorridor_6e51e_00005',
#                  'resume': True,
#                  'reinit': False,
#                  'allow_val_change': False,
#                  'group': 'PPO',
#                  'project': 'Optimization_Project'


# class LogProxyCallback:
# __init__(Proxy)
# def on_trial_result()
#   self.proxy.get_logs()

# class WandbProxy:
    # def __init__(proxy)



from ray.tune.result import TRIAL_INFO
import copy



#print(dir(PPOTrainer))
#print(PPOTrainer.trial_id)
#print(PPOTrainer.trial_name)
#time.sleep(100)

#         print(config["__trial_info__"]._trial_id)
#         config["env_config"]["trial_id"] = config["__trial_info__"]._trial_id
#
# class PPOTrainerv2(PPOTrainer):
#     def __init__(self, **kwargs):
#         print(kwargs["config"]["__trial_info__"])
#         kwargs["config"]["env_config"]["trial_id"] = kwargs["config"]["__trial_info__"].trial_id
#         kwargs["config"]["env_config"]["trial_name"] = kwargs["config"]["__trial_info__"].trial_name
#         PPOTrainer.__init__(self,**kwargs)


    # def _create_logger(self, config, logger_creator):
    #     #print(config.keys())
    #     #print(self.__dict__.keys())
    #     #print(self.config.keys(), config.keys())
    #     time.sleep(100)
    #     self.config["env_config"]["trial_id"] = self.trial_id
    #     PPOTrainer._create_logger(self, config, logger_creator)

#    def __init__(self, config=None, logger_creator=None):
        #trial_info = config.pop(TRIAL_INFO, None)
#        config = copy.deepcopy(config)
#        config["env_config"]["_tr  ial_name"] = config[TRIAL_INFO]._trial_name
#        config["env_config"]["_trial_id"] = config[TRIAL_INFO]._trial_id

        #return super().__init__(self, config, logger_creator)

        #print(trial_info.__dict__.keys())
        #print(config)
        #time.sleep(100)

        #trial_info = self.config.pop(TRIAL_INFO, None)
        #print(trial_info)
        #PPOTrainer.__init__(config, logger_creator)

    #def _init(self, **kwargs):
    #    raise Exception
        #print("init is called")
        #self._init(**kwargs)

    #def setup(self,config):
        #raise Exception
        #print("setup is called")
        #return PPOTrainer.setup(config)
        #print(self.workers)
        #print(PPOTrainer.__dict__.keys())
        #print(self.__dict__.keys())
        #self.config["env_config"]["trial_id"] = self.trial_id

        #time.sleep(100)
        #return super(PPOTrainerv2, self).setup(self.config)

#from optparse import OptionParser
#import inspect
#python2
#print(inspect.getmembers(OptionParser))
#time.sleep(100)




























# class TrialName:
    # config_name + time.strftime("%Y-%m-%d_%H-%M-%S")
    # ___call__()
    # return self.config_name

# `group``, ``run_id`` and ``run_name`


# from ray.tune import Callback
#
#
# class MyCallback(Callback):
#     def on_trial_start(self, iteration, trials, trial, **info):
#         print("on trial start:", trial)
#         time.sleep(100)

# wandb_config = {'id': '6e51e_00005',
#                 'name': 'PPO_SimpleCorridor_6e51e_00005',
#                  'resume': True,
#                  'reinit': False,
#                  'allow_val_change': False,
#                  'group': 'PPO',
#                  'project': 'Optimization_Project',
#                 }
#
# #wandb.init(**wandb_config)
# #id='6e51e_00000', name="some_name", project='Optimization_Project', group="PPO", resume=True)
#
#
# _set_api_key(api_key_file="/home/maksym/Summaries/wandb_key")
# wandb.init(**wandb_config)
#
#
# for i in range(10):
#     wandb.log({"loss": random.random()*i,})



#def myfunc(config):
#    print("111")




from typing import Dict, List

import json
import os

from ray.tune.logger import LoggerCallback

#
# def trainable(config):
#     # config (dict): A dict of hyperparameters.
#
#     for x in range(20):
#         intermediate_score = random.random()
#
#         tune.report(score=intermediate_score, sosos=1)  # This sends the score to Tune.
#
# analysis = tune.run(
#     trainable,
#     config={"a": 2, "b": 4}
# )
#
# print("best config: ", analysis.get_best_config(metric="score", mode="max"))
# print(analysis.__dict__)
#




# wandb.init(project="wandb_rlbo", )
# WandbLoggerCallback(project="Optimization_Project", api_key_file="/path/to/file", log_config=False)
# config {TrialNameCreator:trialName}
# proxy_config: TrialName: trialName()
# 'training_iteration': 1,
# 'timestamp': 1613764232,
# 'time_this_iter_s': 9.550260782241821,
# 'time_total_s': 9.550260782241821,
# 'time_since_restore': 9.550260782241821,
# 'timesteps_since_restore': 0,
# 'iterations_since_restore': 1,

# trial_id: fb9b5_00000
# log trial PPO_SimpleCorridor_fb9b5_00000


#
# def train(config):
#     for i in range(10):
#         tune.report(metric=i)
#
#
# tune.run(
#     train,
#     callbacks=[MyCallback()])