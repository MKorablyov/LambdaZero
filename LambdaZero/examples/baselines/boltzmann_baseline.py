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

import ray
from ray import util


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
