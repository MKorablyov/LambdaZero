import numpy as np
import multiprocessing
import os.path as osp
from rdkit import Chem
from scipy import special
import LambdaZero.chem
import LambdaZero.utils
from LambdaZero.environments import BlockMolEnv_v3, BlockMolEnv_v4
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as env_config

import ray
from ray import util

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
