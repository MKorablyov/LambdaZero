import pandas as pd
import glob
import torch
import torch
import numpy as np
import os
from copy import deepcopy
import shortuuid
import time
from matplotlib import pyplot as plt
import multiprocessing
import time

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 600)
from LambdaZero.contrib.functional import elu2

from LambdaZero.contrib.config_acquirer import oracle_config

EPS = 0.0000004
LOG_CANDIDATE_QED_SCORE = 0.3
LOG_CANDIDATE_SYNTH_SCORE = 4.

DIFF_R_DECIMAL = 3


def true_r_f(dff):
    true_r = dff["dockscore"] * dff['clip_synth'] * dff["clip_qed"]
    return true_r


def get_env_data(env_data_paths):
    if not isinstance(env_data_paths, list):
        env_data_paths = [env_data_paths]

    # env_data_paths = glob.glob(f"{env_data_dir}/env_data*")
    # print(f"Collecting from {len(env_data_paths)} seeds")

    # if os.path.isfile(tmpfile):
    #     env_data = np.load(tmpfile, allow_pickle=True).item()
    #     print(f"LOADED FROM TMP FILE {len(env_data)} smiles")
    # else:
    #     env_data = dict()

    env_data = dict()
    env_smiles = []
    env_duplicates = []
    for ipth, pth in enumerate(env_data_paths):
        try:
            data = torch.load(pth)
        except:
            print(f"error opening {pth}")
            continue

        data["source_file"] = os.path.basename(pth)
        smi = data["smiles"]
        if smi in env_data:
            continue

        # if ipth % 100 == 0:
        #     print(f"Reached {ipth}/{len(env_data_paths)}")
        #     np.save(tmpfile, env_data, allow_pickle=True)
        #     print("SAve checkpoint")

        if smi in env_data:
            env_duplicates.append(data)
            continue
        env_smiles.append(smi)

        env_data[smi] = data
        # ==========================================================================================
        # ==========================================================================================
        # env_data[smi]["obs"]["action_mask"] = env_data[smi]["obs"]["action_mask"].astype(np.bool)
        # mol_graph = env_data[smi]["obs"]["mol_graph"]
        # print(env_data[smi]["obs"].keys())
        # # env_data[smi]["obs"]["mol_graph"] = mol_graph[:np.where(mol_graph != 0)[0].max() + 1]
        # # env_data[smi]["obs"].pop("mol_graph")
        # # env_data[smi]["obs"].pop("action_mask")
        # env_data[smi].pop("obs")
        # ==========================================================================================
        # Filter only necessary info
        next_states = env_data[smi].pop("next_state")
        need_nexts = []
        for next_state in next_states:
            need = dict({
                x: next_state[x] for x in ["info_log_synth_score",
                                           "info_log_qed_score",
                                           "action",
                                           "dockscore",
                                           "mol_smiles",
                                           "true_dockscore"
                                           ]
            })
            need_nexts.append(need)

        # ==========================================================================================
        df = pd.DataFrame(need_nexts)

        df = df.rename(columns={
            "mol_smiles": "next_smiles",
            "info_log_synth_score": "synth_score",
            "info_log_qed_score": "qed_score",
        })

        # ==========================================================================================
        # df["true_r"] = df["dockscore"]

        df["candidate"] = ((df.qed_score > LOG_CANDIDATE_QED_SCORE) &
                           (df.synth_score > LOG_CANDIDATE_SYNTH_SCORE))

        # import pdb; pdb.set_trace()
        env_data[smi]["next"] = df


        # df["true_r_group"] = df.groupby(np.around(df.true_r, DIFF_R_DECIMAL)).ngroup()
        #
        # cand_scores = df.dockscore.copy() if use_proxy_for_cand else df.true_dockscore.copy()
        # cand_scores[~df.candidate] = 333
        # df["candidate_dockscore"] = cand_scores
        # df["candidate_dockscore_group"] = df.groupby(cand_scores).ngroup()
        #
        # assert len(df) == df.action.nunique()
        # df.set_index("action", inplace=True)
        # env_data[smi]["next"] = df
        #
        # # df.sort_values(by="true_r")["true_r_group"].values[::-1]
        # env_data[smi]["target_r_groups"] = df.sort_values(by="true_r")["true_r_group"].values[::-1]
        # env_data[smi]["target_candidate_groups"] = \
        #     df.sort_values(by="candidate_dockscore")["candidate_dockscore_group"].values

        # keep = ['env_seed', 'test_actions', 'jbonds', 'smiles', 'blockidxs', 'obs', "source_file"]
        # rem_k = [x for x in env_data[smi].keys() if x not in keep]
        # print("rem keys", rem_k)
        # for rk in rem_k:
        #     env_data[smi].pop(rk)

    # print(f"Got {len(env_data)} molecules")
    return env_data


def eval_save_data(env_data):
    data = dict()

    for k, v in env_data.items():
        next = v["next"]
        next["next_smiles"] = next["mol_smiles"]
        save_df = next[[
            "true_r", "true_dockscore", "qed_score", "synth_score",
            "candidate", "candidate_dockscore_group", "true_r_group", "next_smiles"
        ]]

        data[k] = dict({
            "next": save_df.copy(),
            "obs": v["obs"],
            'action_mask': v["obs"]["action_mask"].astype(np.bool),
            "env_seed": v["env_seed"],
            "target_r_groups": v["target_r_groups"],
            "target_candidate_groups": v["target_candidate_groups"],
        })

    return data


if __name__ == "__main__":
    import argparse
    import LambdaZero

    datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("env_data_path", type=str)
    parser.add_argument("out_name", type=str)
    parser.add_argument("--tmpload", type=str, default=None)
    args = parser.parse_args()

    tmp = f"{datasets_dir}/tmp_{args.out_name}.npy"

    env_data_paths = glob.glob(f"{args.env_data_path}/env_data*")
    print(f"Collecting from {len(env_data_paths)} seeds")

    # env_data = get_env_data(env_data_paths, tmp)
    p = multiprocessing.Pool(40)
    bsize = 1000
    merge = dict()

    # Load tmp
    if args.tmpload is not None:
        merge = np.load(args.tmpload, allow_pickle=True).item()
        done_fls = [x["source_file"] for x in merge.values()]
        remaining_fls = [x for x in env_data_paths if os.path.basename(x) not in done_fls]
        print(f"Remaining to collect {len(remaining_fls)} seeds")

    for ibtch, batch in enumerate(range(0, len(env_data_paths), bsize)):
        env_data = p.map(get_env_data, env_data_paths[batch: batch + bsize])

        for x in env_data:
            merge.update(x)

        # Save tmp
        save_path = None
        if ibtch % 10 == 0:
            save_path = f"{datasets_dir}/tmp_{args.out_name}.npy"
            np.save(save_path, merge)

        print(f"Done {batch}/{len(env_data_paths)} @ {save_path}")

    print("MERGING")
    env_data = merge

    print("Got data. Calculating eval data")
    # eval_dat = eval_save_data(env_data)
    save_path = f"{datasets_dir}/{args.out_name}.npy"
    print(f"Saving eval_data @ {save_path}")
    np.save(save_path, env_data)