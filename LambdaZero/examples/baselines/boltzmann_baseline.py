import numpy as np
import sys
import os.path as osp
from rdkit import Chem
from scipy import special
import LambdaZero.chem
import LambdaZero.utils
from LambdaZero.examples.baselines import config

import ray
from ray import tune
from ray.rllib.utils import merge_dicts
# import multiprocessing
# from ray import util

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
out_dir = osp.join(summaries_dir, "dock_eval")

# # for dock6 # # principally, could put in config as well
# dock6_dir = osp.join(programs_dir, "dock6")
# chimera_dir = osp.join(programs_dir, "chimera")
# docksetup_dir = osp.join(datasets_dir, "brutal_dock/d4/docksetup")

# for smina
mgltools_dir=osp.join(programs_dir, "mgltools_x86_64Linux2_1.5.6")
vina_dir=osp.join(programs_dir, "vina")
docksetup_dir=osp.join(datasets_dir, "seh/4jnc")

class EnvEval:
    def __init__(self, env, config):
        self.env = env
        # self.dock_smi = LambdaZero.chem.Dock_smi(outpath=out_dir,
        #                                     chimera_dir=chimera_dir,
        #                                     dock6_dir=dock6_dir,
        #                                     docksetup_dir=osp.join(datasets_dir, "brutal_dock/mpro_6lze/docksetup"),
        #                                     gas_charge=True)
        self.dock_smi = LambdaZero.chem.DockVina_smi(outpath=out_dir,
                 mgltools_dir=mgltools_dir,
                 vina_dir=vina_dir,
                 docksetup_dir=docksetup_dir)
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


#@ray.remote
class Boltzmann_opt(tune.Trainable):
    def setup(self, config):
        config["env_config"]["reward_config"]["device"] = "cpu"
        self.env = config["env"](config["env_config"])

        self.temperature = config["temperature"]
        self.steps = config["steps"]
        self.docking = config["docking"]
        self.env_eval = EnvEval(self.env, config["env_eval_config"])

    def reset_setup(self, config):
        self.steps = config["steps"]
        self.docking = config["docking"]
        return True

    def enumerate_actions(self):
        state = self.env.get_state()
        actions = np.where(self.obs["action_mask"])[0]
        values = []
        for i, a in enumerate(actions):
            temp_obs, reward, _, info = self.env.step(a)
            values.append(self.env_eval(docking=self.docking))
            self.env.set_state(state)
        probs = special.softmax(np.divide(values, self.temperature))
        a = np.random.choice(actions.shape[0], p=probs)
        return actions[a], values[a]

    def optimize_molecule(self):  # __call__(self):
        # values = []
        # observations = []
        self.obs = self.env.reset()
        for i in range(self.steps):
            self.step()
            # action, value = self.enumerate_actions()
            # self.obs, reward, _, info = self.env.step(action)
            # # or evaluate the value of the chosen action only
            # # value = self.env_eval(docking=False)
            # values.append(value)
            # # observations.append(obs)
        # return values
        # return None

    def step(self):
        action, value = self.enumerate_actions()
        self.obs, reward, _, info = self.env.step(action)
        log_vals = {**{"reward":value, "molecule":Chem.MolToSmiles(info["molecule"].mol)}, **info["log_vals"]}
        tune.report(log_vals)
        # return log_vals

def boltzmann_opt_wrapper(config):
    boltzmann_opt = Boltzmann_opt(config)
    boltzmann_opt.optimize_molecule()

DEFAULT_CONFIG = {
    "summaries_dir": summaries_dir,
    "memory": 60 * 10 ** 9,
    "reuse_actors": True,
    "num_samples": 200,
    "resources_per_trial": {
        "cpu": 4,
        "gpu": 0.4,
        # usually, request 40 cpus, 4 gpus (entire node on Beluga)
    },
}

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    else:
        config_name = "boltzmann_config_001"
    config = getattr(config, config_name)
    config = merge_dicts(DEFAULT_CONFIG, config)

    ray.init(memory=config["memory"])
    analysis=tune.run(
        Boltzmann_opt,
        config=config["boltzmann_config"],
        num_samples=config["num_samples"],
        reuse_actors=config["reuse_actors"],
        resources_per_trial=config["resources_per_trial"],
        local_dir=config["summaries_dir"],
    )

    # # for a single actor:
    # actor = Boltzmann_opt.remote(config)
    # remaining_result_ids = [actor.optimize_molecule.remote() for i in range(10)]
    # results = []
    # while len(remaining_result_ids) > 0:
    #     ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
    #     result_id = ready_result_ids[0]
    #     result = ray.get(result_id)
    #     results.append(result)

    # #
    # # # for an actor pool
    # actors = [Boltzmann_opt.remote(config) for _ in range(multiprocessing.cpu_count())]
    # pool = util.ActorPool(actors)
    # for i in range(10000):
    #     pool.submit(lambda a, v: a.optimize_molecule.remote(), 0)
    #
    # results = []
    # while pool.has_next():
    #     result = pool.get_next_unordered()
    #     print(result)
    #     results.append(result)