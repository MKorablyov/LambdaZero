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
from ray.tune.integration.wandb import WandbLoggerCallback, _set_api_key

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
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1, ), dtype=np.float32)
        self.steps = 0
        print("env initialized!!!!!", config["trial_id"])
    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        self.steps +=1
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos


        return [self.cur_pos], 1.0 if done else -0.1, done, {}



# from ray.tune import Callback
# class MyCallback(Callback):
#     def on_trial_start(self, iteration: int, trials: List["Trial"],trial: "Trial", **info):
#         trials[0].config["env_config"]["trial_id"] = trials[0].trial_id
#         print("trial start callback!!!!")
#         time.sleep(1000)
wandb_logger = WandbLoggerCallback(project="Optimization_Project",
                                   api_key_file="/home/maksym/Summaries/wandb_key",
                                   log_config=False)

class PPOTrainerv2(PPOTrainer):
    def setup(self, config):
        self.config["env_config"]["trial_id"] = self.trial_id
        return super(PPOTrainerv2, self).setup(self.config)


tune.run(
    PPOTrainerv2,
    checkpoint_freq=1,
    config={
        "framework": "torch",
        "num_workers": 7,
        "num_gpus": 1,
        "env": SimpleCorridor,
        "env_config":{"corridor_length":7},

        },
    stop={"training_iteration":1},
    loggers=DEFAULT_LOGGERS + (wandb_logger,),
    #callbacks=[MyCallback()],
)

























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