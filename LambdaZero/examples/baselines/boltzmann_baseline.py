<<<<<<< HEAD
import copy
import functools
import multiprocessing
from os import path
import time

import numpy as np
from scipy import special

from LambdaZero import chem
from LambdaZero.environments import molecule
from LambdaZero.examples import config as cfg

from rdkit import Chem
=======
import numpy as np
import multiprocessing
import os.path as osp
from rdkit import Chem
from scipy import special
import LambdaZero.chem
import LambdaZero.utils
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as env_config
>>>>>>> 1b9837d4af7d97954b7aff8aacee2856f85ff1a1

import ray
from ray import util

<<<<<<< HEAD

def step_env(rng, env, state, action_mask, temperature, evaluate_molecules):
    actions = np.where(action_mask)[0]
    molecules = []
    next_states = []
    observations = []

    for i, a in enumerate(actions):
        env.set_state(state)
        obs, _, _, info = env.step(a)
        molecules.append(copy.deepcopy(info["molecule"].mol))
        next_states.append(env.get_state())
        observations.append(obs)

    values = evaluate_molecules(molecules)
    probs = special.softmax(np.divide(values, temperature))

    a = rng.choice(actions.shape[0], p=probs)
    return values[a], next_states[a], observations[a]


def boltzmann_search(max_steps, mol_eval, total_evals, temperature=1.,
                     stop_condition=None, top_k=1):
    config = cfg.mol_blocks_v4_config()
    config["env_config"]["reward_config"]["device"] = "cpu"
    env = config["env"](config["env_config"])

    obs = env.reset()
    state = env.get_state()

    values = []
    states = []
    rng = np.random.default_rng()

    for i in range(max_steps):
        action_mask = obs["action_mask"]
        val, state, obs = step_env(rng, env, state, action_mask, temperature,
                                   mol_eval)
        values.append(val)
        states.append(state)

        print((
            "Finished iteration {}, current max: {:.3f}, total evals: {}"
        ).format(i, np.max(values), total_evals()))

        if stop_condition is not None:
            stop_condition.update(val, state, obs)
            if stop_condition.should_stop():
                break

    top_idx = np.argsort(values)[-top_k:]
    return tuple(zip(*[(values[i], states[i]) for i in top_idx]))


LAMBDAZERO_ROOT = "/home/gehring/PycharmProjects/LambdaZero/"
PROGRAM_DIR = path.join(LAMBDAZERO_ROOT, "Programs")
DATASET_DIR = path.join(LAMBDAZERO_ROOT, "Datasets")

DOCK_OUT_DIR = path.join(DATASET_DIR, "dock")
DOCK6_DIR = path.join(PROGRAM_DIR, "dock6")
CHIMERA_DIR = path.join(PROGRAM_DIR, "chimera")
DOCKSETUP_DIR = path.join(DATASET_DIR, "brutal_dock/d4/docksetup")


@ray.remote
class DockActor:

    def __init__(self):
        self._dock_smi = chem.Dock_smi(outpath=DOCK_OUT_DIR,
                                       chimera_dir=CHIMERA_DIR,
                                       dock6_dir=DOCK6_DIR,
                                       docksetup_dir=DOCKSETUP_DIR)
        self._num_evals = 0

    def evaluate_molecule(self, molecule):
        try:
            self._num_evals += 1
            _, energy, _ = self._dock_smi.dock(Chem.MolToSmiles(molecule))
        except AssertionError:
            energy = np.inf
        return -energy

    def evaluation_count(self):
        return self._num_evals


if __name__ == "__main__":

    ray.init()
    dockactors = [DockActor.remote() for _ in range(multiprocessing.cpu_count())]
    pool = util.ActorPool(dockactors)

    def evaluate_molecules(molecules):
        return list(
            pool.map(lambda a, m: a.evaluate_molecule.remote(m), molecules))

    def total_evaluations():
        return np.sum(ray.get([a.evaluation_count.remote() for a in dockactors]))

    values, states = boltzmann_search(100, evaluate_molecules, total_evaluations,
                                      1.0, None, 3)
    print(states)
    print(values)
=======
default_config = {
    "env": BlockMolEnv_v4,
    "env_config": env_config,
    "temperature": 1.,
    "steps": 8,
    "env_eval_config":{
        "dockscore_norm": [-43.042, 7.057]}
}


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
out_dir = osp.join(summaries_dir, "dock_eval")
dock6_dir = osp.join(programs_dir, "dock6")
chimera_dir = osp.join(programs_dir, "chimera")
docksetup_dir = osp.join(datasets_dir, "brutal_dock/d4/docksetup")


class EnvEval:
    def __init__(self, env, config):
        self.env = env
        self.dock_smi = LambdaZero.chem.Dock_smi(outpath=out_dir,
                                            chimera_dir=chimera_dir,
                                            dock6_dir=dock6_dir,
                                            docksetup_dir=osp.join(datasets_dir, "brutal_dock/mpro_6lze/docksetup"),
                                            gas_charge=True)
        self.dockscore_norm = config["dockscore_norm"]
        self._num_evals = 0

    def __call__(self, docking=False):
        if docking:
            try:
                smi = Chem.MolToSmiles(self.env.molecule.mol)
                gridscore = self.dock_smi.dock(smi)[1]
                dock_reward = -((gridscore - self.dockscore_norm[0]) / self.dockscore_norm[1])
            except Exception as e:
                dock_reward = None
            return dock_reward
        else:
            return self.env.reward.previous_reward


@ray.remote
class Boltzmann_opt:
    def __init__(self, config):
        config["env_config"]["reward_config"]["device"] = "cpu"
        self.env = config["env"](config["env_config"])
        self.temperature = config["temperature"]
        self.steps = config["steps"]

        self.env_eval = EnvEval(self.env, config["env_eval_config"])

    def enumerate_actions(self, obs):
        state = self.env.get_state()
        actions = np.where(obs["action_mask"])[0]
        values = []
        for i, a in enumerate(actions):
            obs, reward, _, info = self.env.step(a)
            values.append(self.env_eval(docking=False))
            self.env.set_state(state)
        probs = special.softmax(np.divide(values, self.temperature))
        a = np.random.choice(actions.shape[0], p=probs)
        return actions[a], values[a]

    def optimize_molecule(self):  # __call__(self):
        values = []
        # observations = []
        obs = self.env.reset()
        for i in range(self.steps):
            action, value = self.enumerate_actions(obs)
            obs, reward, _, info = self.env.step(action)
            # or evaluate the value of the chosen action only
            # value = self.env_eval(docking=False)
            values.append(value)
            # observations.append(obs)
        return values

if __name__ == "__main__":
    ray.init()
    config = default_config

    # # for a single actor:
    # actor = Boltzmann_opt.remote(config)
    # remaining_result_ids = [actor.optimize_molecule.remote() for i in range(10)]
    # results = []
    # while len(remaining_result_ids) > 0:
    #     ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
    #     result_id = ready_result_ids[0]
    #     result = ray.get(result_id)
    #     results.append(result)


    # # for an actor pool
    actors = [Boltzmann_opt.remote(config) for _ in range(multiprocessing.cpu_count())]
    pool = util.ActorPool(actors)
    for i in range(10000):
        pool.submit(lambda a, v: a.optimize_molecule.remote(), 0)

    results = []
    while pool.has_next():
        result = pool.get_next_unordered()
        print(result)
        results.append(result)
>>>>>>> 1b9837d4af7d97954b7aff8aacee2856f85ff1a1
