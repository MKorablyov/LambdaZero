import copy
import functools
import multiprocessing
import os
from os import path
import time
import subprocess
import pickle
import gzip

import numpy as np
from scipy import special

from LambdaZero import chem
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments.reward import PredDockReward_v3 as PredReward
from LambdaZero.environments.block_mol_v3 import DEFAULT_CONFIG as env_cfg, BlockMolEnv_v3
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

from rdkit import Chem

import ray
from ray import util
from ray.rllib.utils import merge_dicts


def step_env(rng, env, state, action_mask, temperature, evaluate_molecules):
    actions = np.where(action_mask)[0]
    molecules = []
    next_states = []
    observations = []
    smiles = []

    for i, a in enumerate(actions):
        env.set_state(state)
        obs, _, _, info = env.step(a)
        molecules.append(copy.deepcopy(env.molMDP.molecule))
        next_states.append(env.get_state())
        observations.append(obs)
        smiles.append(info['molecule'])

    values_pd = [(v[0], i, j, v[1], sm)
                 for v, (j, i), sm in
                 zip(evaluate_molecules(molecules), enumerate(molecules), smiles)]

    if 0:
        probs = special.softmax(np.divide([i[0] for i in values_pd], temperature))
        a = rng.choice(actions.shape[0], p=probs, size=min(5, int((probs>0).sum())), replace=False)
        for i in [molecules[i].mol for i in a if molecules[i].mol is not None]:
            print(chemprop.forward([Chem.MolToSmiles(i)]))
        return [molecules[i].mol for i in a], [next_states[i] for i in a]
    top_k = sorted(values_pd, key=lambda x:x[0])[-5:]
    return [i[1].mol for i in top_k], [next_states[i[2]] for i in top_k], [i[3] for i in top_k], [i[4] for i in top_k]


def beam_search(max_steps, mol_eval, total_evals, mol_eval_pd, total_evals_pd,
                temperature=1.,
                stop_condition=None, beam_size=20):
    env = BlockMolEnv_v3(env_cfg)


    beam_states = []
    beam_values = []
    beam_logs = []
    for i in range(beam_size//2):
        obs = env.reset()
        state = env.get_state()
        beam_states.append(state)
        beam_values.append(0)
        beam_logs.append({})



    salt = hex(abs(hash(str(time.time()))))[2:6]
    exp_path = time.strftime(f'beam_results_%m%b_%d_%H_%M_{salt}.pkl.gz')
    all_mols = []
    rng = np.random.default_rng()

    for i in range(max_steps):
        new_beam_mols = []
        new_beam_states = []
        new_beam_logs = []
        new_beam_smiles = []
        for state in beam_states:
            obs = env.set_state(state)
            action_mask = obs["action_mask"]
            mols, state, logs, smiles = step_env(rng, env, state, action_mask, temperature,
                                                 mol_eval_pd)
            new_beam_mols += mols
            new_beam_states += state
            new_beam_logs += logs
            new_beam_smiles += smiles
        new_beam_values = mol_eval(new_beam_mols)
        beam_states += new_beam_states
        beam_values += new_beam_values
        beam_logs += new_beam_logs
        for s, v, l, sm in zip(new_beam_states, new_beam_values, new_beam_logs, new_beam_smiles):
            all_mols.append((s, sm, v, l))

        beam_discounts = [i.get('discount', 0) for i in beam_logs]
        probs = special.softmax(np.divide(np.multiply(beam_values, beam_discounts), temperature))
        print(np.sort(probs)[-3:], probs.min(), probs.var())
        bolt = rng.choice(len(probs), p=probs,
                          size=min(beam_size//2, int((probs>0).sum())), replace=False)
        top_idx = np.argsort([v if i not in bolt else -100
                              for i, v in enumerate(beam_values)])[-(beam_size-len(bolt)):]
        top_idx = sorted(list(top_idx) + list(bolt), key=lambda i: probs[i])
        beam_values = [beam_values[i] for i in top_idx]
        beam_states = [beam_states[i] for i in top_idx]
        beam_logs = [beam_logs[i] for i in top_idx]

        print((
            "Finished iteration {}, current max: {:.3f}, total evals: {}, total mpnn calls: {}"
        ).format(i, np.max(beam_values), total_evals(), total_evals_pd()))
        print(' '.join(f"{i:.2f}" for i in beam_values))
        for i, l in zip(beam_values, beam_logs):
            print(f'{i:.2f}', ' '.join([f'{k}={v:.2f}' for k,v in l.items()]))

        if stop_condition is not None:
            stop_condition.update(val, state, obs)
            if stop_condition.should_stop():
                break
        pickle.dump(all_mols, gzip.open(exp_path, 'w'))
    return beam_values, beam_states


if not os.path.exists(os.environ["SLURM_TMPDIR"]+"/Programs"):
    print("Running Programs rsync...")
    subprocess.run(["cp", "-R", programs_dir, os.environ["SLURM_TMPDIR"]])
    print("Done")
programs_dir = os.environ["SLURM_TMPDIR"]+"/Programs"

DOCK_OUT_DIR = os.environ["SLURM_TMPDIR"]
DOCK6_DIR = path.join(programs_dir, "dock6")
CHIMERA_DIR = path.join(programs_dir, "chimera")
DOCKSETUP_DIR = path.join(datasets_dir, "brutal_dock/mpro_6lze/docksetup")#"brutal_dock/d4/docksetup")


@ray.remote
class DockActor:

    def __init__(self):
        self._dock_smi = chem.Dock_smi(outpath=DOCK_OUT_DIR,
                                       chimera_dir=CHIMERA_DIR,
                                       dock6_dir=DOCK6_DIR,
                                       docksetup_dir=DOCKSETUP_DIR,
                                       gas_charge=True)
        self._num_evals = 0
        self.dockscore_std = binding_config["dockscore_std"]

    def evaluate_molecule(self, molecule):
        try:
            self._num_evals += 1
            _, energy, _ = self._dock_smi.dock(Chem.MolToSmiles(molecule))
        except AssertionError:
            energy = np.inf
        energy = (self.dockscore_std[0] - energy) / (self.dockscore_std[1])
        return energy

    def evaluation_count(self):
        return self._num_evals

@ray.remote(num_gpus=0.1)
class PredDockActor:
    def __init__(self):
        reward_config = env_cfg['reward_config']
        self.pred = PredReward(**reward_config)
        self._num_evals = 0

    def evaluate_molecule(self, molecule):
        log_vals = {}
        try:
            self._num_evals += 1
            if molecule is None:
                raise AssertionError()
            self.pred.reset(0)
            energy, log_vals = self.pred(molecule, True, True, 0)
        except AssertionError:
            energy = -np.inf
        except ValueError as e:
            print(e)
            energy = -np.inf
        if energy is None:
            energy = -np.inf
        return energy, log_vals

    def evaluation_count(self):
        return self._num_evals




if __name__ == "__main__":
    env_cfg = merge_dicts(
        env_cfg,
        {
            "random_steps": 0,
            "allow_removal": True,
            "reward": PredReward,
            "reward_config": {
                "synth_config": synth_config,
                "dockscore_config": binding_config,
        }})

    ray.init(num_cpus=16, ignore_reinit_error=True)
    dockactors = [DockActor.remote() for _ in range(8)]#multiprocessing.cpu_count())]
    pool = util.ActorPool(dockactors)

    def evaluate_molecules(molecules):
        return list(
            pool.map(lambda a, m: a.evaluate_molecule.remote(m), molecules))

    def total_evaluations():
        return np.sum(ray.get([a.evaluation_count.remote() for a in dockactors]))

    preddockactors = [PredDockActor.remote() for _ in range(8)]#multiprocessing.cpu_count())]
    predpool = util.ActorPool(preddockactors)

    def evaluate_molecules_pd(molecules):
        return list(
            predpool.map(lambda a, m: a.evaluate_molecule.remote(m), molecules))

    def total_evaluations_pd():
        return np.sum(ray.get([a.evaluation_count.remote() for a in preddockactors]))


    values, states = beam_search(100, evaluate_molecules, total_evaluations,
                                 evaluate_molecules_pd, total_evaluations_pd,
                                 2.0, None, 50)

    print(states)
    print(values)
