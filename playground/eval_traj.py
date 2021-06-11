import copy

from LambdaZero.contrib.config_rlbo import trainer_config
from LambdaZero.environments import BlockMolEnvGraph_v1
from ray.rllib.models.preprocessors import get_preprocessor
import ray
from LambdaZero.contrib.loggers import RemoteLogger
from LambdaZero.contrib.oracle.oracle import DockingEstimator
from LambdaZero.contrib.config_acquirer import oracle_config
import numpy as np
import time
import base64
import csv
import pickle
import shortuuid
from actor_pool_wait import ActorPoolWait

import sys

fld_out = "/home/andrein/scratch/analysis"

luid = shortuuid.uuid()
lname = f"{int(time.time())}_{luid}"
data_out = f"{fld_out}/data_{lname}.csv"

# old_stdout = sys.stdout
# log_file = open(f"{fld_out}/log_{lname}.log","w")
# sys.stdout = log_file

# ---- Remote score proxy
ray.init()

# -- Load proxy
config = trainer_config
config["env_config"]['random_steps'] = 1

remote_logger = RemoteLogger.remote()
scP_cfg = config['env_config']["reward_config"]["scoreProxy_config"]
r_cfg = config['env_config']['reward_config']
scP_cfg["logger"] = remote_logger
scP_cfg["oracle_config"]["logger"] = remote_logger
scP_cfg["acquirer_config"]["model_config"]["logger"] = remote_logger

scoreProxy = r_cfg['scoreProxy'].options(**r_cfg['scoreProxy_options']).remote(**r_cfg['scoreProxy_config'])
r_cfg['scoreProxy'] = scoreProxy

env_config = config["env_config"]

env = BlockMolEnvGraph_v1(env_config)
# -----


def reset_env(env, random_steps, rnd_state=None):
    env._reset_mode = True  # Signal reset mode

    env.num_steps = 0
    env.molMDP.reset()
    env.reward.reset()
    obs, graph = env._make_obs()

    prev_action = []

    for i in range(random_steps):
        actions = np.where(obs["action_mask"])[0]
        action = np.random.choice(actions) if rnd_state is None else rnd_state.choice(actions)
        prev_action.append(action)
        env.step(action)
        obs, graph = env._make_obs()
        if env._if_terminate():
            print("bad molecule init: resetting MDP")
            env.molMDP.reset()
    try:
        assert env.molMDP.molecule is not None, "molecule is None"
        # try if the molecule produces valid smiles string
        env.molMDP.molecule.smiles
    except Exception as e:
        print("initialized environment with invalid molecule", e)
        return env.reset()
    env.num_steps = 0

    obs["prev_actions"] = prev_action
    env._reset_mode = False  # Signal reset mode

    return obs


def get_mol_data(env):
    molmdp_data = {"blockidxs": env.molMDP.molecule.blockidxs,
                   # "blocks": env.molMDP.molecule.blocks,
                   "slices": env.molMDP.molecule.slices,
                   "numblocks": env.molMDP.molecule.numblocks,
                   "jbonds": env.molMDP.molecule.jbonds,
                   "stems": env.molMDP.molecule.stems,
                   "smiles": env.molMDP.molecule.smiles}
    return molmdp_data

num_threads_dock = 25
dockVina_config = oracle_config["dockVina_config"]

dock_actors = [DockingEstimator.remote(dockVina_config) for i in range(num_threads_dock)]
# dock_pool = ray.util.ActorPool(dock_actors)
dock_pool = ActorPoolWait(dock_actors, timeout=240)




# env = BlockMolEnvGraph_v1({"random_steps": 1, "allow_removal": True})
# space = env.graph_mol_obs.space
# prep = get_preprocessor(env.observation_space)(env.observation_space)

env_resets = [(1, None), (1, 11), (2, 14), (4, 16)]

# =========================
from LambdaZero import chem
from rdkit import Chem

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

env = BlockMolEnvGraph_v1({"random_steps": 1, "allow_removal": True})
env.max_steps = 30
env.reward = RewardDummy()

wrong_idxs = []
for i, row in dft.iterrows():
    blocks = [env.molMDP.block_mols[x] for x in row["mol_data_blockidxs"]]
    try:
        mol, _ = chem.mol_from_frag(jun_bonds=row["mol_data_jbonds"], frags=blocks)
        smi = Chem.MolToSmiles(mol)
    except:
        wrong_idxs.append(i)
        continue
    if row["mol_data_smiles"] != smi:
        wrong_idxs.append(i)

    if i % 1000 == 0:
        print(i, len(wrong_idxs))


def get_copy_mol_data(env):
    molmdp_data = copy.deepcopy({"mol_data_blockidxs": env.molMDP.molecule.blockidxs,
                                 "mol_data_slices": env.molMDP.molecule.slices,
                                 "mol_data_numblocks": env.molMDP.molecule.numblocks,
                                 "mol_data_jbonds": env.molMDP.molecule.jbonds,
                                 "mol_data_stems": env.molMDP.molecule.stems,
                                 "mol_data_smiles": env.molMDP.molecule.smiles})
    return molmdp_data

new_data = dict({})

groups = dft.groupby(["file_id", "tr_cnt"])
group_cnt = len(groups)
igroup  = 0
for ig, group in groups:
    sgroup = group.sort_values("iobs")
    prev_act = sgroup.iloc[0]["obs_prev_actions"]

    new_obs = reset_clean(env)
    env._reset_mode = True  # Signal reset mode
    for act in prev_act:
        obs, r, done, info = env.step(act)
    test = env.molMDP.molecule.smiles
    obs, graph = env._make_obs()

    env.num_steps = 0
    env._reset_mode = False  # Signal reset mode

    if env.molMDP.molecule.smiles != sgroup.iloc[0]["mol_data_smiles"]:
        print("BAD :(", ig, 0)
        break

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
            print("No act :(", ig, irow)
            break

        obs, r, done, info = env.step(found)

        if env.molMDP.molecule.smiles != next_smi:
            print("BAD :(", ig, irow)
            break

        new_data[sgroup.iloc[irow].name] = get_copy_mol_data(env)
    print(f"Finished: {igroup}/{group_cnt}")
    igroup+=1

obs = reset_env(env, 1, rnd_state=np.random.RandomState(11))
print(env.molMDP.molecule.smiles)
steps = df[(df.file_id == "data_1617806472_coMsPr7eYSfZSAtRxiGtew") & (df.tr_cnt == 2)].iloc[:]["mol_data_smiles"].values

obs, r, done, info = env.step(29)
actions = np.where(obs["action_mask"])[0]

step = 3


for act in actions:
    cpenv = copy.deepcopy(env)
    cpenv.step(act)
    if steps[step] == cpenv.molMDP.molecule.smiles:
        print("found", step, act)
        break

true_steps = [50, 148, 29]
print(len(env.molMDP.molecule.jbonds) != len(env.molMDP.molecule.blockidxs)-1)
action = np.random.choice(actions)
# ===================
import ray
ray.init()

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
        test = env.molMDP.molecule.smiles
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


num_actors = 1
pool = ray.util.ActorPool([TestEnv.remote(env) for _ in range(num_actors)])

groups = list(dft.groupby(["file_id", "tr_cnt"]))
group_cnt = len(groups)

send_arg = [groups[2]]

return_p = list(pool.map(lambda a, v: a.test.remote(*v), send_arg))

# ===================

# ===================
iobs = 0
tr_cnt = 0

with open(data_out, 'a', encoding='utf8') as csv_file:
    wr = csv.writer(csv_file, delimiter='|')

    def write_obj(obj):
        pickle_bytes = pickle.dumps(obj)            # unsafe to write
        b64_bytes = base64.b64encode(pickle_bytes)  # safe to write but still bytes
        b64_str = b64_bytes.decode('utf8')          # safe and in utf8
        wr.writerow([b64_str])

    while True:
        smiles = []
        states = []
        tr_times = []

        for group, (rnd_steps, rnd_state) in enumerate(env_resets):
            rnds = None if rnd_state is None else np.random.RandomState(rnd_state)
            tr_cnt += 1
            tr_time = time.time()

            obs = reset_env(env, rnd_steps, rnd_state=rnds)
            smiles.append(env.molMDP.molecule.smiles)

            st = time.time()
            eval_obs = env.reward.eval(env.molMDP.molecule)
            tp_eval = time.time() - st
            mol_data = get_mol_data(env)
            iobs += 1

            obs["action_mask"] = obs["action_mask"].astype(np.bool)

            data = dict({"obs": obs, "eval": eval_obs, "r": None, "done": False, "info": None,
                         "tp_eval": tp_eval, "iobs": iobs, "tr_cnt": tr_cnt, "env_group": group,
                         "mol_data": mol_data})
            states.append(data)

            done = False
            while not done:

                actions = np.where(obs["action_mask"])[0]
                action = np.random.choice(actions)

                obs, reward, done, info = env.step(action)
                print(group, done)

                iobs += 1

                smiles.append(env.molMDP.molecule.smiles)

                st = time.time()
                eval_obs = env.reward.eval(env.molMDP.molecule)

                tp_eval = time.time() - st
                mol_data = get_mol_data(env)

                obs["action_mask"] = obs["action_mask"].astype(np.bool)
                obs["prev_actions"] = [action]
                data = dict({"obs": obs, "eval": eval_obs, "r": reward, "done": done, "info": info,
                             "tp_eval": tp_eval, "iobs": iobs, "tr_cnt": tr_cnt, "env_group": group,
                             "mol_data": mol_data})
                states.append(data)

                tr_time = time.time() - tr_time
                tr_times.append(tr_time)

        print(f"START DOCK SCORE for... {len(smiles)} smiles")
        st = time.time()
        dockscores = list(dock_pool.map(lambda actor, smi: actor.eval.remote(smi), smiles))
        dock_time = time.time() - st

        print(f"{dock_time} write")
        # import pdb; pdb.set_trace()
        write_obj((states, dockscores, tr_times))


