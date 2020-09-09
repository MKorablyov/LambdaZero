import copy
import functools
import socket, multiprocessing, time, os.path as osp
from os import path
import time

import numpy as np
from scipy import special

import LambdaZero
from LambdaZero.environments import reward
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4, BlockMolEnv_v5, PredDockReward, PredDockReward_v2
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as env_config
from rdkit import Chem

import ray
#from ray import util

default_config = {
    "env": BlockMolEnv_v4,
    "env_config": env_config
}

def boltzmann_choice(probs, actions):
    a = np.random.choice(actions, p=probs)
    return a

def enumerate_actions(env, obs, temperature):
    state = env.get_state()
    actions = np.where(obs["action_mask"])[0]
    rewards = []
    for i, a in enumerate(actions):
        obs, reward, _, info = env.step(a)
        rewards.append(reward)
        env.set_state(state)
    probs = special.softmax(np.divide(rewards, temperature))
    return actions, probs

@ray.remote
def boltzmann_opt(env, temperature=2., steps=10):
    rewards = []
    observations = []
    obs = env.reset()
    for i in range(steps):
        actions, probs = enumerate_actions(env, obs, temperature)
        #print ('done')
        action = boltzmann_choice(probs, actions)
        obs, reward, _, info = env.step(action)
        rewards.append(reward)
        observations.append(obs)
    return max(rewards)

# @ray.remote
# class boltzmann_opt:
#
#     def __init__(self, env, temperature=1., steps=10):
#         self.env = env
#         self.temperature = temperature
#         self.steps = steps
#
#     def boltzmann_choice(self, probs, actions):
#         a = np.random.choice(actions, p=probs)
#         return a
#
#     def enumerate_actions(self, obs):
#         state = self.env.get_state()
#         actions = np.where(obs["action_mask"])[0]
#         rewards = []
#         for i, a in enumerate(actions):
#             obs, reward, _, info = env.step(a)
#             rewards.append(reward)
#             env.set_state(state)
#         probs = special.softmax(np.divide(rewards, self.temperature))
#         return actions, probs
#
#     def __call__(self):
#         rewards = []
#         observations = []
#         obs = env.reset()
#         for i in range(self.steps):
#             actions, probs = self.enumerate_actions(obs)
#             action = self.boltzmann_choice(probs, actions)
#             obs, reward, _, info = env.step(action)
#             rewards.append(reward)
#             observations.append(obs)
#         return max(rewards)


if __name__ == "__main__":

    ray.init()
    config = default_config
    config["env_config"]["reward_config"]["device"] = "cpu"
    env = config["env"](config["env_config"])
    reward = ray.get(boltzmann_opt.remote(env))
    print (reward)
    #value = ray.get(obj)
    #print("highest value is", value)


#
# def step_env(rng, env, state, action_mask, temperature):
#     actions = np.where(action_mask)[0]
#     # molecules = []
#     next_states = []
#     observations = []
#     values = []
#
#     for i, a in enumerate(actions):
#         env.set_state(state)
#         obs, reward, _, info = env.step(a)
#         #molecules.append(copy.deepcopy(info["molecule"].mol))
#         next_states.append(env.get_state())
#         observations.append(obs)
#         values.append(reward)
#     #values = evaluate_molecules(molecules)
#     probs = special.softmax(np.divide(values, temperature))
#     a = rng.choice(actions.shape[0], p=probs)
#     return values[a], next_states[a], observations[a]
#
# @ray.remote
# def boltzmann_search(env, state, obs, max_steps, temperature=1.):
#     values = []
#     # states = []
#     rng = np.random.default_rng()
#     for i in range(max_steps):
#         action_mask = obs["action_mask"]
#         val, state, obs = step_env(rng, env, state, action_mask, temperature)
#         values.append(val)
#         #states.append(state)
#         print(("Finished iteration {}, current max: {:.3f}").format(i, np.max(values)))
#     return max(values)
#     # top_idx = np.argsort(values)[-top_k:]
#     # return tuple(zip(*[(values[i], states[i]) for i in top_idx]))

# if __name__ == "__main__":
#     config = default_config
#     config["env_config"]["reward_config"]["device"] = "cpu"
#     env = config["env"](config["env_config"])
#     times = 0
#     for i in range(10000):
#         obs = env.reset()
#         state = env.get_state()
#         # values, states = boltzmann_search(env, state, obs, 8, evaluate_molecules, total_evaluations, 1.0, None, 1)
#         obj = boltzmann_search.remote(env, state, obs, 12,
#                                       evaluate_molecules,
#                                           1.0, None, 1)
#         # times += timesz
#         # print(states)
#         value = ray.get(obj)
#         print("highest value is", value)
#     print (times/100)







# datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
# DATASET_DIR, PROGRAM_DIR, _ = LambdaZero.utils.get_external_dirs()
#
# DOCK_OUT_DIR = path.join(DATASET_DIR, "dock")
# DOCK6_DIR = path.join(PROGRAM_DIR, "dock6")
# CHIMERA_DIR = path.join(PROGRAM_DIR, "chimera")
# DOCKSETUP_DIR = path.join(DATASET_DIR, "brutal_dock/d4/docksetup")
#
#
# @ray.remote
# class DockActor:
#
#     def __init__(self):
#         self._dock_smi = chem.Dock_smi(outpath=DOCK_OUT_DIR,
#                                        chimera_dir=CHIMERA_DIR,
#                                        dock6_dir=DOCK6_DIR,
#                                        docksetup_dir=DOCKSETUP_DIR)
#         self._num_evals = 0
#
#     def evaluate_molecule(self, molecule):
#         try:
#             self._num_evals += 1
#             _, energy, _ = self._dock_smi.dock(Chem.MolToSmiles(molecule))
#         except AssertionError:
#             energy = np.inf
#         return -energy
#
#     def evaluation_count(self):
#         return self._num_evals
#
#
# if __name__ == "__main__":
#
#     ray.init()
#     dockactors = [DockActor.remote() for _ in range(multiprocessing.cpu_count())]
#     pool = util.ActorPool(dockactors)
#     def evaluate_molecules(molecules):
#         return list(
#             pool.map(lambda a, m: a.evaluate_molecule.remote(m), molecules))
#
#     def total_evaluations():
#         return np.sum(ray.get([a.evaluation_count.remote() for a in dockactors]))
#
#     values, states = boltzmann_search(100, evaluate_molecules, total_evaluations,
#                                       1.0, None, 3)
#     print(states)
#     print(values)
