import numpy as np
from scipy import special

from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as env_config

import ray
#from ray import util

default_config = {
    "env": BlockMolEnv_v4,
    "env_config": env_config,
    "temperature": 2.,
    "steps": 8
}

# def boltzmann_choice(probs, actions):
#     a = np.random.choice(actions, p=probs)
#     return a
#
# def enumerate_actions(env, obs, temperature):
#     state = env.get_state()
#     actions = np.where(obs["action_mask"])[0]
#     rewards = []
#     for i, a in enumerate(actions):
#         obs, reward, _, info = env.step(a)
#         rewards.append(reward)
#         env.set_state(state)
#     probs = special.softmax(np.divide(rewards, temperature))
#     return actions, probs
#
# @ray.remote
# def boltzmann_opt(env, temperature=2., steps=10):
#     rewards = []
#     observations = []
#     obs = env.reset()
#     for i in range(steps):
#         actions, probs = enumerate_actions(env, obs, temperature)
#         #print ('done')
#         action = boltzmann_choice(probs, actions)
#         obs, reward, _, info = env.step(action)
#         rewards.append(reward)
#         observations.append(obs)
#     return max(rewards)

@ray.remote
class boltzmann_opt:

    def __init__(self, config):
        config["env_config"]["reward_config"]["device"] = "cpu"
        self.env = config["env"](config["env_config"])
        self.temperature = config["temperature"]
        self.steps = config["steps"]

    def boltzmann_choice(self, probs, actions):
        action = np.random.choice(actions, p=probs)
        return action

    def enumerate_actions(self, obs):
        state = self.env.get_state()
        actions = np.where(obs["action_mask"])[0]
        rewards = []
        for i, a in enumerate(actions):
            obs, reward, _, info = self.env.step(a)
            rewards.append(reward)
            self.env.set_state(state)
        probs = special.softmax(np.divide(rewards, self.temperature))
        return actions, probs

    def optimize(self):  # __call__(self):
        rewards = []
        # observations = []
        obs = self.env.reset()
        for i in range(self.steps):
            actions, probs = self.enumerate_actions(obs)
            action = self.boltzmann_choice(probs, actions)
            obs, reward, _, info = self.env.step(action)
            rewards.append(reward)
            # observations.append(obs)
        return rewards

@ray.remote
def caller():
    for i in range(10000):
        rewards = ray.get(optimizer.optimize.remote())
        print(rewards)

if __name__ == "__main__":
    ray.init()
    config = default_config
    optimizer = boltzmann_opt.remote(config)
    ray.get(caller.remote())
    #optimizer = boltzmann_opt(config)

