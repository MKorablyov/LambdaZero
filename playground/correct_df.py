import numpy as np
from copy import deepcopy
import pandas as pd
import copy
from LambdaZero.environments import BlockMolEnvGraph_v1
from LambdaZero import chem
from rdkit import Chem
import ray
import torch

dft = pd.read_pickle("data/trajectories/all_data.pk")

ray.init()


def get_copy_mol_data(env):
    molmdp_data = copy.deepcopy({"mol_data_blockidxs": env.molMDP.molecule.blockidxs,
                                 "mol_data_slices": env.molMDP.molecule.slices,
                                 "mol_data_numblocks": env.molMDP.molecule.numblocks,
                                 "mol_data_jbonds": env.molMDP.molecule.jbonds,
                                 "mol_data_stems": env.molMDP.molecule.stems,
                                 "mol_data_smiles": env.molMDP.molecule.smiles})
    return molmdp_data


class RewardDummy:
    def __init__(self):
        pass

    def reset(self, previous_reward=0.0):
        pass

    def __call__(self, molecule, simulate, env_stop, num_steps):
        return 0, {}


def reset_clean(env):
    env._reset_mode = True  # Signal reset mode

    env.num_steps = 0
    env.molMDP.reset()
    env.reward.reset()
    obs, graph = env._make_obs()

    env.num_steps = 0
    env._reset_mode = False  # Signal reset mode


@ray.remote
class TestEnv(object):
    def __init__(self, env):
        self.env = env

    def test(self, ig, group):
        print(ig, len(group))
        new_data = dict()
        bad = None
        env = self.env
        sgroup = group.sort_values("iobs")
        prev_act = sgroup.iloc[0]["obs_prev_actions"]

        new_obs = reset_clean(env)
        env._reset_mode = True  # Signal reset mode
        for act in prev_act:
            obs, r, done, info = env.step(act)
        try:
            assert env.molMDP.molecule is not None, "molecule is None"
            # try if the molecule produces valid smiles string
            test = env.molMDP.molecule.smiles
        except Exception as e:
            print("initialized environment with invalid molecule", e)
            return new_data, bad

        obs, graph = env._make_obs()

        env.num_steps = 0
        env._reset_mode = False  # Signal reset mode

        if env.molMDP.molecule.smiles != sgroup.iloc[0]["mol_data_smiles"]:
            print("BAD :(", 0)
            bad = 0
            return new_data, bad

        new_data[sgroup.iloc[0].name] = get_copy_mol_data(env)

        for irow in range(1, len(sgroup)):
            next_smi = sgroup.iloc[irow]["mol_data_smiles"]

            actions = np.where(obs["action_mask"])[0]
            found = None
            for act in actions:
                cpenv = copy.deepcopy(env)
                cobs, creward, cdone, cinfo = cpenv.step(act)
                if next_smi == cpenv.molMDP.molecule.smiles:
                    found = act
                    break

            if found is None:
                print("No act :(", irow)
                bad = irow
                return new_data, bad

            obs, r, done, info = env.step(found)

            if env.molMDP.molecule.smiles != next_smi:
                print("BAD :(", irow)
                bad = irow
                return new_data, bad

            new_data[sgroup.iloc[irow].name] = get_copy_mol_data(env)
        return new_data, bad


env = BlockMolEnvGraph_v1({"random_steps": 1, "allow_removal": True})
env.max_steps = 30
del env.reward
env.reward = RewardDummy()


num_actors = 40
pool = ray.util.ActorPool([TestEnv.remote(env) for _ in range(num_actors)])

groups = list(dft.groupby(["file_id", "tr_cnt"]))
group_cnt = len(groups)


all_p = []
st = 9567
end = 18887 + 1
for i in range(st, end, num_actors):
    print(f"send {i}: {i + num_actors} / {group_cnt}")
    send_arg = groups[i:i + num_actors]

    return_p = list(pool.map(lambda a, v: a.test.remote(*v), send_arg))
    all_p += return_p

    torch.save(all_p, f"/home/andrein/scratch/tmp/new_data_{st}_{end}")