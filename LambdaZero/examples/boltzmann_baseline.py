import copy
import functools
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


def boltzmann_search(max_steps, mol_eval, temperature=1., stop_condition=None,
                     top_k=1):
    config = cfg.mol_blocks_v4_config()
    config["env_config"]["reward_config"]["device"] = "cpu"
    env = config["env"](config["env_config"])

    obs = env.reset()
    state = env.get_state()

    values = []
    states = []
    rng = np.random.default_rng()

    for i in range(max_steps):
        print(i)
        action_mask = obs["action_mask"]
        val, state, obs = step_env(rng, env, state, action_mask, temperature,
                                   mol_eval)
        values.append(val)
        states.append(state)

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

    def evaluate_molecule(self, molecule):
        _, energy, _ = self._dock_smi.dock(Chem.MolToSmiles(molecule))
        return energy


if __name__ == "__main__":

    # rew_eval = molecule.PredDockReward(
    #     load_model=path.join(DATASET_DIR, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
    #     natm_cutoff=[45, 50],
    #     qed_cutoff=[0.2, 0.7],
    #     soft_stop=False,
    #     exp=None,
    #     delta=False,
    #     simulation_cost=0.0,
    #     device="cpu",
    # )
    #
    # def eval_molecule(mol):
    #     rew_eval.reset()
    #     return rew_eval(mol,
    #                     env_stop=False,
    #                     simulate=True,
    #                     num_steps=1)[0]
    # evaluate random molecules with docking

    ray.init(local_mode=False)
    pool = util.ActorPool([DockActor.remote() for _ in range(24)])

    def evaluate_molecules(molecules):
        return list(
            pool.map(lambda a, m: a.evaluate_molecule.remote(m), molecules))

    values, states = boltzmann_search(100, evaluate_molecules, 1.0, None, 3)
    print(states)
    print(values)
