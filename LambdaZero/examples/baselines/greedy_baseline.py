from LambdaZero.environments import chemMDP, MolMDP, reward, BlockMolEnv_v4
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as env_config
import LambdaZero
import time
import numpy as np
import ray
from rdkit import Chem
from copy import copy, deepcopy
#import concurrent.futures
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

default_config = {
    "env": BlockMolEnv_v4,
    "env_config": env_config,
    "steps": 10
}

@ray.remote
class greedy_opt:

    def __init__(self, config):
        config["env_config"]["reward_config"]["device"] = "cpu"
        self.env = config["env"](config["env_config"])
        self.steps = config["steps"]

    def enumerate_actions(self, obs):
        state = self.env.get_state()
        actions = np.where(obs["action_mask"])[0]
        prev_reward = -100.
        for i, a in enumerate(actions):
            obs, reward, _, info = self.env.step(a)
            if prev_reward < reward:
                prev_reward = deepcopy(reward)
                action = deepcopy(a)
            self.env.set_state(state)
        return action

    def optimize(self):  # __call__(self):
        rewards = []
        # observations = []
        obs = self.env.reset()
        for i in range(self.steps):
            action = self.enumerate_actions(obs)
            obs, reward, _, info = self.env.step(action)
            # print (reward)
            rewards.append(reward)
            # observations.append(obs)
        return rewards

if __name__ == "__main__":
    ray.init()
    config = default_config
    optimizer = greedy_opt.remote(config)
    rewards = ray.get(optimizer.optimize.remote())
    print(rewards)


#optimizer = boltzmann_opt(config)
# @ray.remote
# def caller():
#     ray.get(caller.remote())