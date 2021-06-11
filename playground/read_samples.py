import csv
import pickle
import base64
import argparse
import pprint
import  os
import csv
import numpy as np
from copy import deepcopy
import pandas as pd
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import FormatStrFormatter
from LambdaZero.contrib.config_model import load_seen_config
# plt.interactive(False)
# from eval_traj import get_mol_data
from LambdaZero.environments import BlockMolEnvGraph_v1

# matplotlib.use('TkAgg')
import ray
from ray.util import ActorPool

# ray.init()

def merge_data():
    dir_path = "data/trajectories/raw_data/"
    data_paths = glob.glob(f"{dir_path}/*.csv")
    # data_paths = ["/home/andrein/scratch/analysis/data_1617896537_QYHMiVuEHdBBSZxbYShaG4.csv"]

    all_data = []
    for iii, data_path in enumerate(data_paths):
        print(f"CHECK file {iii}/{len(data_paths)} ({data_path})")
        path = data_path
        data = read_csv(data_path)
        all_data += data

    df = pd.DataFrame.from_dict(all_data)
    # df.to_feather("data/trajectories/all_data.feather")
    df.to_pickle("data/trajectories/all_data.pk")
    #
    # store = pd.HDFStore("data/trajectories/all_data.h5")
    #
    # store['df'] = df  # save it
    #
    # dft = pd.read_pickle("data/trajectories/all_data.pk")
    #[, , , , , ]
    # dfsmiles = dft[["mol_data_smiles", "dock_score"]]
    #
    # CLean
    clms = ['blockidxs', 'slices', 'jbonds', 'stems']
    # dfu['blockidxs'] = dfu['blockidxs'].apply(str)
    # (dfu['blockidxs'].apply(lambda x: "array" in x)).sum()
    #
    # dfu['slices'] = dfu['slices'].apply(str)
    # (dfu['slices'].apply(lambda x: "array" in x)).sum()
    #
    # dfu['jbonds'] = dfu['jbonds'].apply(str)
    # new_jb = dfu['jbonds'].apply(lambda x: str(eval(x.replace("array", "list"))) if "array" in x else x)
    # (new_jb.apply(lambda x: "array" in x)).sum()
    # dfu['jbonds'] = new_jb
    #
    # dfu['stems'] = dfu['stems'].apply(str)
    # (dfu['stems'].apply(lambda x: "array" in x)).sum()
    # new_stems = dfu['stems'].apply(lambda x: str(eval(x.replace("array", "list"))) if "array" in x else x)
    # (new_stems.apply(lambda x: "array" in x)).sum()
    # dfu['stems'] = new_stems

def rename_dict(d: dict, prefix: str):
    keys = list(d.keys())
    for k in keys:
        d[f"{prefix}_{k}"] = d.pop(k)
    return d


def get_mol_data(env):
    molmdp_data = {"blockidxs": env.molMDP.molecule.blockidxs,
                   # "blocks": env.molMDP.molecule.blocks,
                   "slices": env.molMDP.molecule.slices,
                   "numblocks": env.molMDP.molecule.numblocks,
                   "jbonds": env.molMDP.molecule.jbonds,
                   "stems": env.molMDP.molecule.stems,
                   "smiles": env.molMDP.molecule.smiles}
    return molmdp_data


class FakeR:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, molecule, simulate, env_stop, num_steps):
        return 0.0, {}

    def reset(self, previous_reward=0.0):
        pass


def reset_empty():
    env = BlockMolEnvGraph_v1({"random_steps": 1, "allow_removal": True, "reward": FakeR})
    env.num_steps = 0
    env.molMDP.reset()
    env.reward.reset()
    obs, graph = env._make_obs()
    return env


def detect_action(env, obs, next_obs):
    act_mask = obs["action_mask"]
    actions = np.where(obs["action_mask"])[0]
    pacts = []
    for pact in actions:
        new_env = deepcopy(env)
        pobs, reward, done, info = new_env.step(pact)
        if np.all(next_obs["mol_graph"] == pobs["mol_graph"]):
            pacts.append(pact)
            return pacts
    return pacts


@ray.remote
class TestEnv(object):
    def __init__(self):
        self.value = 0

    def test(self, env, pact, next_obs):
        new_env = deepcopy(env)
        pacts = []

        pobs, reward, done, info = new_env.step(pact)
        if np.all(next_obs["mol_graph"] == pobs["mol_graph"]):
            pacts.append(pact)
            return pacts
        return []

# num_actors = 10
# pool = ActorPool([TestEnv.remote() for _ in range(num_actors)])


def detect_action_p(env, obs, next_obs):
    act_mask = obs["action_mask"]
    actions = np.where(obs["action_mask"])[0]
    pacts = []

    for i in range(0, len(actions), num_actors):
        send_arg = [[env, pact, next_obs] for pact in actions[i:i + num_actors]]
        return_p = pool.map(lambda a, v: a.test.remote(*v), send_arg)
        for ppp in return_p:
            pacts += ppp

        if len(pacts) > 0:
            return pacts
    return pacts


def read_csv(path: str):
    rows = []
    data_out = f"data/trajectories/fix2/{os.path.basename(path)}"
    # data_out = f"/home/andrein/scratch/fix2/{os.path.basename(path)}"
    data_name = os.path.splitext(os.path.basename(path))[0]

    if os.path.isfile(data_out):
        path = data_out

    with open(path, 'r') as csv_file:
        for line in csv_file:
            line = line.strip('\n')
            b64_str = line # .split('|')[0]  # 1 column
            obj = pickle.loads(base64.b64decode(b64_str))  # retrieve
            rows.append(obj)

    all_rows = []
    if not os.path.isfile(data_out):
        for row in rows:
            if len(row[0]) != len(row[1]):
                print("diff", data_out)
            if len(row) == 2:
                all_rows += list(zip(*row))
            elif len(row) == 3:
                all_rows += list(zip(*row[:2]))

        rows = all_rows

    # check if needs fix (missing molecule data)
    obs = rows[0][0]

    if "mol_data" not in obs.keys():
        # FIX
        env_empty = reset_empty()

        done = True

        for istep, tr_next in enumerate(rows):
            print(f"{istep}/{len(rows)} {done}")

            if done:
                env = deepcopy(env_empty)
                start_actions = tr_next[0]["obs"]["prev_actions"]

                for act in start_actions:
                    obs, reward, done, info = env.step(act)

                env.num_steps = 0

                pre_obs = tr_next[0]["obs"]
                assert np.all(obs["mol_graph"] == pre_obs["mol_graph"]), "NOT same obs"
                assert np.all(obs["action_mask"] == pre_obs["action_mask"]), "NOT same obs"

                mol_data = get_mol_data(env)
                tr_next[0]["mol_data"] = mol_data
                continue

            # tr_next = trajectory_data[1]
            # break
            next_obs = tr_next[0]["obs"]

            # pact = detect_action(env, pre_obs, next_obs)
            pact = detect_action_p(env, pre_obs, next_obs)
            assert len(pact) == 1, "What no action?!??"

            obs, reward, done, info = env.step(pact[0])
            assert np.all(obs["mol_graph"] == next_obs["mol_graph"]), "NOT same obs"
            assert np.all(obs["action_mask"] == next_obs["action_mask"]), "NOT same obs"

            mol_data = get_mol_data(env)
            tr_next[0]["mol_data"] = mol_data

            pre_obs = next_obs

        with open(data_out, 'a', encoding='utf8') as csv_file:

            wr = csv.writer(csv_file, delimiter='|')

            def write_obj(obj):
                pickle_bytes = pickle.dumps(obj)  # unsafe to write
                b64_bytes = base64.b64encode(pickle_bytes)  # safe to write but still bytes
                b64_str = b64_bytes.decode('utf8')  # safe and in utf8
                wr.writerow([b64_str])

            for itr, trajectory_data in enumerate(rows):
                write_obj(trajectory_data)

    # Align dock score
    all_obs = []
    for i, r in enumerate(rows):
        if len(r) == 2:
            obs, dock_score = r
            tr_time = np.nan
        else:
            obs, dock_score, tr_time = r
        obs["dock_score"] = dock_score
        obs["row"] = i
        obs["tr_time"] = tr_time
        all_obs.append(obs)

    # Flatten info in dict
    all_data = []
    for state in all_obs:
        data = dict({"file_id": data_name})
        obs = rename_dict(state.pop("obs"), "obs")

        eval_r, eval_info = state.pop('eval')
        eval_info["r"] = eval_r
        eval_info = rename_dict(eval_info, "eval")

        if "mol_data" not in state:
            print("Fix it")
            break

        mol_data = rename_dict(state.pop("mol_data"), "mol_data")

        data.update(obs)
        data.update(eval_info)
        data.update(mol_data)
        data.update(state)
        all_data.append(data)

    return all_data

def analyse(all_data):

    pprint.pprint(all_data[0])

    df = pd.DataFrame.from_dict(all_data)

    dock_mean, dock_std = load_seen_config["mean"], load_seen_config["std"]

    df["norm_dock"] = df.dock_score.copy()
    cap_filter = df.norm_dock > (dock_mean + 3 * dock_std)
    print(f"{sum(cap_filter)/len(cap_filter)} are capped!!!")
    df.norm_dock[cap_filter] = dock_mean + 3 * dock_std

    df.norm_dock = (dock_mean - df.norm_dock) / dock_std


    dfp = df
    flt_na = ~df["dock_score"].isna()
    flt_cand = (df["eval_qed_score"] > 0.5) & (df["eval_synth_score"] > 4)
    dfp = df[flt_na & flt_cand]



    dfp.dock_score.plot(kind="box")

    dfp.plot.scatter("dock_score", "qed")
    dfp.plot.scatter("dock_score", "eval_qed_score")
    dfp.plot.scatter("norm_dock", "eval_proxy_dock_mean")

    # explained variance
    import sklearn
    flt_na = ~df["dock_score"].isna()
    ytrue = df["dock_score"][flt_na]
    ypred = df["norm_dock"][flt_na]
    sklearn.metrics.explained_variance_score(ytrue, ypred)


    df.groupby("smiles")

def analys():
    import glob
    import pandas as pd
    import torch
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    from matplotlib.ticker import FormatStrFormatter
    from LambdaZero.contrib.config_model import load_seen_config

    # exp = "rlbo_001_rsteps1"
    # exp = "rlbo_001"
    exp = "rlbo_001_30k"
    dir_path = f"/media/andrei/CE04D7C504D7AF292/milaintern/lz_branch/data/debug2/{exp}_"
    type = "acq_model_pred_1"

    acq_pths = glob.glob(f"{dir_path}acq*")
    unacq_pths = glob.glob(f"{dir_path}unacq*")

    data = []
    for i in range(len(acq_pths)):
        data_acq = torch.load(f"{dir_path}acq_model_pred_{i}")
        for k in list(data_acq.keys()):
            data_acq[f"acq_{k}"] = data_acq.pop(k)

        data_unacq = torch.load(f"{dir_path}unacq_model_pred__{i}")
        data_acq.update(data_unacq)
        data.append(data_acq)

    df = pd.DataFrame.from_dict(data)

    acq_mean = np.concatenate(df["acq_before_mean"].values)
    acq_mean_after = np.concatenate(df["acq_after_mean"].values)
    acq_tgt = np.concatenate(df["acq_target"].values)

    una_mean = np.concatenate(df["unacq_before_mean"].values)
    una_mean_after = np.concatenate(df["unacq_after_mean"].values)
    una_tgt = np.concatenate(df["unacq_target"].values)

    fig1, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].set_title("Before model.fit")
    ax[0].scatter(una_mean, una_tgt, label="unacquired_sample")
    ax[0].scatter(acq_mean, acq_tgt, label="acquired_batch")
    ax[0].set_xlabel('predicted score')
    ax[0].set_ylabel('oracle score')
    ax[0].set_xlim(-7.5, 5)
    ax[0].set_ylim(-4, 5)
    ax[0].legend()

    ax[1].set_title("After model.fit")
    ax[1].scatter(una_mean_after, una_tgt, label="unacquired_sample")
    ax[1].scatter(acq_mean_after, acq_tgt, label="acquired_batch")
    ax[1].set_xlabel('predicted score')
    ax[1].set_ylabel('oracle score')
    ax[1].legend()
    ax[1].set_xlim(-7.5, 5)
    ax[1].set_ylim(-4, 5)
    plt.suptitle(f"Experiment {exp}")
    plt.show()
    plt.savefig(f"data/{exp}.png")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process traj logs.')
    # parser.add_argument('paths', metavar='N', type=str, nargs='+',
    #                     help='an integer for the accumulator')
    # args = parser.parse_args()
    #
    # read_csv(args.paths[0])
    merge_data()