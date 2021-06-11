import sqlite3
from sqlite3 import Error
import socket
from copy import deepcopy as copy
import numpy as np

from LambdaZero.environments import BlockMolEnvGraph_v1


def reset_env(env, random_steps, rnd_state=None):
    """ Reset env deterministically """
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
        assert not env._if_terminate(), "bad molecule init: resetting MDP"

    assert env.molMDP.molecule is not None, "molecule is None"
    env.num_steps = 0

    obs["prev_actions"] = prev_action
    return obs


class DummyReward:
    def __init__(self, binding_model, qed_cutoff, synth_cutoff, synth_config,
                 soft_stop, exp, delta, simulation_cost, device):
        pass

    def reset(self, previous_reward=0.0):
        pass

    def _discount(self, mol, reward):
        return 0, {"dock_reward": 0, "natm": 0, "qed" : 0, "synth" : 0}

    def _simulation(self, molecule):
        return 0

    def __call__(self, molecule, simulate, env_stop, num_steps):
        return 0, {}


env = BlockMolEnvGraph_v1({"random_steps": 1, "reward": DummyReward})

obs = reset_env(env, 2, rnd_state=np.random.RandomState(30))


action_mask = obs["action_mask"]
action_mask[:7] = 0

history = [[copy(env), action_mask]]
level_options = [0] * 3


future_act = []
pact = np.where(history[-1][1])[0]
while len(pact) > 0:
    lvl = len(history)
    next_env = copy(history[-1][0])
    obs, reward, done, info = next_env.step(pact[0])
    if not done:
        num_next = len(np.where(obs["action_mask"])[0])
        level_options[lvl] += num_next

    # Remove steps done
    history[-1][1][pact[0]] = 0

    action_mask = obs["action_mask"]
    action_mask[:7] = 0

    if len(history) < 2 and not done:
        history.append([copy(next_env), action_mask])
    elif len(pact) == 1:
        history.pop(-1)

    pact = np.where(history[-1][1])[0]

    if len(history) == 1:
        print(len(pact))

actions =

num_act = []
if len(actions) > 0:
    act = actions[0]
    action_mask[act] = 0






def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def test_lmdb():
    import pandas as pd
    import lmdb
    import numpy as np
    import timeit
    import ast, json

    df = pd.read_pickle("data/trajectories/all_for_db.p")
    idxs = df.index
    env = lmdb.open("data/trajectories/lmdb_test", map_size=int(1e9))
    txn=env.begin(write=True)

    dtp = df.iloc[i].values.tolist()
    dtp = [eval(x) for x in dtp[:-1]] + [dtp[-1]]
    dtp2 = [str(dtp[:-1]), dtp[-1]]
    timeit.timeit("str(dtp)", globals=dict(dtp=dtp), number=100000)
    timeit.timeit("str(dtp).encode('ascii')", globals=dict(dtp=dtp), number=100000)
    timeit.timeit("eval(str(dtp).encode('ascii').decode('ascii'))", globals=dict(dtp=dtp, ast=ast), number=100000)
    timeit.timeit("json.loads(str(dtp2).encode('ascii').decode('ascii'))", globals=dict(dtp2=dtp2, json=json), number=100000)
    #
    # timeit.timeit("str(-1.2).encode('ascii')", globals=dict(dtp=dtp), number=100000)
    # timeit.timeit("str(-1.2).encode('ascii').decode('ascii')", globals=dict(dtp=dtp), number=100000)
    # timeit.timeit("float(str(-1.2).encode('ascii').decode('ascii'))", globals=dict(dtp=dtp), number=100000)
    #
    # %%timeit str(dtp).encode('ascii').decode('ascii')

    for i in range(len(df)):
        txn.put(df.iloc[i].name.encode('ascii'), df.iloc[i].values.tostring())
        if i % 1000 == 0:
            print(i)

    txn.commit()
    env.close()

    # Read
    env = lmdb.open("data/trajectories/lmdb_test")
    # Create transaction
    txn = env.begin()
    # Query a single record
    dp = txn.get(np.random.choice(idxs).encode('ascii'))
    StringIO(dp)
    print()  # b "mark"
    # b "2" b "bob"
    # b "3" b "mark"
    for key, value in txn.cursor():
        print(key, value)
    # Commit changes
    txn.commit()

    if __name__ == '__main__':
    machine = socket.gethostname()
    if machine == "andrei":
        pth_db = "data/lambdazero.db"
    else:
        pth_db = r"/home/andrein/scratch/Summaries/lambdazero.db"

    create_connection(r"/home/andrein/scratch/Summaries/lambdazero.db")
