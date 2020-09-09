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

def step_env(rng, env, state, action_mask, temperature, evaluate_molecules):
    actions = np.where(action_mask)[0]
    # molecules = []
    next_states = []
    observations = []
    values = []

    for i, a in enumerate(actions):
        env.set_state(state)
        obs, reward, _, info = env.step(a)
        #molecules.append(copy.deepcopy(info["molecule"].mol))
        next_states.append(env.get_state())
        observations.append(obs)
        values.append(reward)
    #values = evaluate_molecules(molecules)
    probs = special.softmax(np.divide(values, temperature))
    a = rng.choice(actions.shape[0], p=probs)
    return values[a], next_states[a], observations[a]

@ray.remote
def boltzmann_search(env, state, obs, max_steps, mol_eval, temperature=1., stop_condition=None, top_k=1):
    # start = time.time()
    values = []
    # states = []
    rng = np.random.default_rng()
    for i in range(max_steps):
        action_mask = obs["action_mask"]
        val, state, obs = step_env(rng, env, state, action_mask, temperature, mol_eval)
        values.append(val)
        #states.append(state)
        print(("Finished iteration {}, current max: {:.3f}").format(i, np.max(values)))
        # print((
        #     "Finished iteration {}, current max: {:.3f}, total evals: {}"
        # ).format(i, np.max(values), total_evals()))
        if stop_condition is not None:
            stop_condition.update(val, state, obs)
            if stop_condition.should_stop():
                break

    # end = time.time()
    # time = end-start
    # print("time", end - start)
    return max(values)

    # top_idx = np.argsort(values)[-top_k:]
    # return tuple(zip(*[(values[i], states[i]) for i in top_idx]))


# def boltzmann_choice(probs, temperature)
    # return sample

# def enumerate_actions(env):
    # return actions, probs

# @ray.remote
# def boltzmann_opt(env, n):
    # for i in rage(n):
        # env.reset()
        # actions, values = enumerate_actions
        # action = boltzmann_choice(probs, temperature)
        # env.step(action)
        # return energy


#  ray.wait_for()
    #


if __name__ == "__main__":

    ray.init()
    # dockactors = [DockActor.remote() for _ in range(multiprocessing.cpu_count())]
    # pool = util.ActorPool(dockactors)
    reward_func = reward.PredDockReward_v2(default_config['env_config']["reward_config"]['binding_model'],
                                           default_config['env_config']["reward_config"]['qed_cutoff'],
                                           default_config['env_config']["reward_config"]['synth_cutoff'],
                                           default_config['env_config']["reward_config"]['synth_config'],
                                           default_config['env_config']["reward_config"]['soft_stop'],
                                           default_config['env_config']["reward_config"]['exp'],
                                           default_config['env_config']["reward_config"]['delta'],
                                           default_config['env_config']["reward_config"]['simulation_cost'],
                                           default_config['env_config']["reward_config"]['device'])

    def evaluate_molecules(molecules):
        rewards = []
        for molecule in molecules:
            dock = reward_func._simulation(molecule)
            if dock is not None:
                discounted_reward, log_val = reward_func._discount(molecule, dock)
                rewards.append(discounted_reward)
        return rewards
        # return list(
        #    pool.map(lambda a, m: a.evaluate_molecule.remote(m), molecules))


    config = default_config
    config["env_config"]["reward_config"]["device"] = "cpu"
    env = config["env"](config["env_config"])
    times = 0
    for i in range(10000):
        obs = env.reset()
        state = env.get_state()
        # values, states = boltzmann_search(env, state, obs, 8, evaluate_molecules, total_evaluations, 1.0, None, 1)
        obj = boltzmann_search.remote(env, state, obs, 12,
                                      evaluate_molecules,
                                          1.0, None, 1)
        # times += timesz
        # print(states)
        value = ray.get(obj)
        print("highest value is", value)
    print (times/100)



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
