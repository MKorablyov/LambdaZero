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


# def EnvEval(env):
    # def __call__()
    # either docking or just use observation (should work with docking and also with the natural reward function)
    # return score


# @ray.remote
# class Actor:
    # def init(config):
    # EnvCreator, env_config,  env_eval, env_eval_config, tempt
    # env_eval = env_eval(env_eval_config)
    # env = self.EnvCreator(env_config)

    # def optimize_molecule()
        # obs = env.reset()

        # for i in range(self.num_steps):
            # state1 = env.get_state()

            # for a in np.where(obs["action_mask"]
                # obs = env.step()
                # value = self.env_eval(env)
                # env.set_state(state1)
            # do boltzman choose aciton
            # obs, reward, _, info = self.env.step(action)



# crate a pool of 4 or 8 actors
# dockactors = [Actor.remote() for _ in range(8)]
# pool.map



# pool = util.ActorPool(dockactors)
# tasks = [a.optimize_molecule.remote() for i in range(16)]


#for i in range(10000):
    # result, tasks  = ray.wait(tasks,1)
    # todo: how to get actor that finished
    # tasks.append(a.optimize_molecule.remote())


# dockactors = [DockActor.remote()
# https://github.com/ray-project/tutorial/blob/master/exercises/exercise05-Actor_Handles.ipynb
# https://github.com/ray-project/tutorial/blob/master/exercises/exercise06-Wait.ipynb
# todo : figure out how to do report as it goes


# [ for i in range(10000)]
# ray.wait_for
# map 10,000 molecules



@ray.remote
class Boltzmann_opt:
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



if __name__ == "__main__":
    ray.init()
    config = default_config
    optimizer = Boltzmann_opt.remote(config)
    rewards = ray.get(optimizer.optimize.remote())
    print(rewards)

