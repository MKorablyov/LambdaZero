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

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 600)
from LambdaZero.contrib.functional import elu2

from LambdaZero.contrib.config_acquirer import oracle_config

EPS = 0.0000004
LOG_CANDIDATE_QED_SCORE = 0.35
LOG_CANDIDATE_SYNTH_SCORE = 4.

DIFF_R_DECIMAL = 3

def get_env_start_state():
    """
    ([108], [66])
    ([130], [173])
    ([180], [145])
    """
    from LambdaZero.contrib.config_rlbo import DEFAULT_CONFIG
    from LambdaZero.contrib.loggers import RemoteLogger
    from LambdaZero.contrib import config_rlbo
    from LambdaZero.environments.block_mol_graph_v1 import BlockMolEnvGraph_Eval
    from ray.rllib.utils import merge_dicts
    import ray

    env_data = dict()

    ray.init()

    config_name = "rlbo_env"

    def get_dir_name(config, experiment_tag=""):
        generated_dirname = f"test_{config_name}_{experiment_tag}_{shortuuid.uuid()}"
        return generated_dirname

    config = getattr(config_rlbo, config_name)
    config = orig_cfg = merge_dicts(DEFAULT_CONFIG, config)
    config = config["tune_config"]['config']

    remote_logger = RemoteLogger.remote()

    remote_logger.set_config.remote(config_name)
    local_dir = os.path.join(orig_cfg["tune_config"]["local_dir"], get_dir_name(config))
    orig_cfg["tune_config"]["local_dir"] = local_dir
    os.mkdir(local_dir)
    remote_logger.set_log_dir.remote(local_dir)
    orig_cfg["tune_config"]['config']['env_config']["logger"] = remote_logger

    scP_cfg = config['env_config']["reward_config"]["scoreProxy_config"]
    r_cfg = config['env_config']['reward_config']
    scP_cfg["logger"] = remote_logger
    scP_cfg["oracle_config"]["logger"] = remote_logger
    scP_cfg["acquirer_config"]["model_config"]["logger"] = remote_logger

    scoreProxy = r_cfg['scoreProxy'].options(**r_cfg['scoreProxy_options']).remote(**r_cfg['scoreProxy_config'])
    r_cfg['scoreProxy'] = scoreProxy

    env_config = config["env_config"]
    # ======================================================================================
    orig_seeds_to_test = deepcopy(env_config['env_seed'])
    print(f"RUNNING FOR seeds: {orig_seeds_to_test}")
    print(f"PATH: {local_dir}")

    env = BlockMolEnvGraph_Eval(env_config)
    env_config = env._config
    orig_seeds_to_test = list(range(13, 213))
    for i, env_seed in enumerate(orig_seeds_to_test):
        st = time.time()
        env_config["env_seed"] = [env_seed]
        env.set_seed(env_config)

        obs = env.reset()
        test_actions = np.where(obs["action_mask"])[0]

        info_all = dict({
            "obs": deepcopy(obs),
            "test_actions": deepcopy(test_actions),
            "jbonds": deepcopy(env.molMDP.molecule.jbonds),
            "smiles": deepcopy(env.molMDP.molecule.smiles),
            "blockidxs": env.molMDP.molecule.blockidxs,
        })
        env_data[env_seed] = info_all
    return env_data


def true_r_f(dff):
    true_r = dff["dockscore"] * dff['clip_synth'] * dff["clip_qed"]
    return true_r


def get_env_data():
    env_data_dir = "data/env_data"
    env_data_paths = glob.glob(f"{env_data_dir}/env_data*")
    env_data = dict()
    env_smiles = []
    env_duplicates = []
    for pth in env_data_paths:
        data = torch.load(pth)
        smi = data["smiles"]

        if smi in env_data:
            env_duplicates.append(data)
            continue
        env_smiles.append(smi)


        env_data[smi] = data
        # env_data[smi]["next_state"][0]
        df = pd.DataFrame(env_data[smi]["next_state"])

        cl_r = dict()
        for cl in df.columns:
            if "info_log_" in cl:
                new_cl = cl.replace("info_log_", "")
                if new_cl not in df.columns:
                    cl_r[cl] = new_cl
        df = df.rename(cl_r, axis=1)

        # TODO !!!!!!THIS IS because we do not log true dockscore
        # true_r = df["dockscore"].copy()  # Because of log bug
        # df["dockscore"] = oracle_config["mean"] - (df['dockscore'] * oracle_config["std"])

        # TODO Calculate The True R that was trained with -
        # TODO Consider mean and norm and act
        # df["true_r"] = true_r * df['clip_synth'] * df["clip_qed"]  # true_r_f(df)
        # df["dockscore"] = df["dockscore"].apply(lambda x: float(np.float16(x)))
        df["true_r"] = true_r_f(df)  # true_r_f(df)

        inv_dock = df.dockscore.min() - df.dockscore
        inv_dock = inv_dock / inv_dock.sum()
        df["inv_norm_dockscore"] = inv_dock

        df["candidate"] = ((df.qed_score > LOG_CANDIDATE_QED_SCORE) &
                           (df.synth_score > LOG_CANDIDATE_SYNTH_SCORE))

        df["true_r_group"] = df.groupby(np.around(df.true_r, DIFF_R_DECIMAL)).ngroup()

        cand_scores = df.true_dockscore.copy()
        cand_scores[~df.candidate] = 333
        df["candidate_dockscore"] = cand_scores
        df["candidate_dockscore_group"] = df.groupby(cand_scores).ngroup()


        assert len(df) == df.action.nunique()
        df.set_index("action", inplace=True)
        env_data[smi]["next"] = df

        # df.sort_values(by="true_r")["true_r_group"].values[::-1]
        env_data[smi]["target_r_groups"] = df.sort_values(by="true_r")["true_r_group"].values[::-1]
        env_data[smi]["target_candidate_groups"] = \
            df.sort_values(by="candidate_dockscore")["candidate_dockscore_group"].values

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

    # np.save("data/full_env_data", env_data)
    np.save("data/eval_env_data", data)

# ==================================================================================================
# Check Target groups
non_zero_cand = []
zero_cand = []
for k, v in env_data.items():
    # num_act_act = (v["obs"]["action_mask"]).sum()
    num_cand = v["next"]["candidate"].sum()
    num_non_zero_r = (v["next"]["true_r"] > 0).sum()
    # print(f"{k} | Num act {num_act_act} | Num R > 0: {num_non_zero_r} | Num cand: {num_cand}")
    if num_cand > 0:
        non_zero_cand.append(v["env_seed"][0])
    else:
        zero_cand.append(v["env_seed"][0])

all_seeds = np.concatenate([zero_cand, non_zero_cand])
all_seeds.sort()
with_cand_seeds = np.ones_like(all_seeds).astype(np.bool)
for x in zero_cand:
    with_cand_seeds[np.where(all_seeds==x)[0][0]] = False

# ==================================================================================================

env_data = get_env_data()

# exp_path = "data/1step/rlbo_eval_50_PPO"
# exp_path = "data/1step/rlbo_eval_100_PPO"
# exp_path = "data/1step/rlbo_eval_50_PPO"
# exp_path = "data/1step/rlbo_eval_test_50"
# exp_path = "data/1step/rlbo_eval_test_50_e_PPO"
# exp_path = "data/1step/rlbo_eval_test_50_long"
# exp_path = "data/1step/rlbo_eval_test_50_highe"
# exp_path = "data/1step/rlbo_eval_test_50_highe2"
exp_path = "data/1step/rlbo_eval_debug_50_long"
# exp_path = "data/1step/rlbo_eval_debug_50"
# exp_path = "data/1step/rlbo_eval_debug_50_egnn"
# exp_path = "data/1step/rlbo_eval_test_50_mlp"
# exp_path = "data/1step/exp5_50_mlp_2"
eval_data_p = glob.glob(f"{exp_path}/eval_data/batch*")


exp_data = []
all_actions_r = []
for i, e_d_p in enumerate(eval_data_p):
    datap = torch.load(e_d_p)

    # Check a bit the exper ment
    # Same training stept f r all eval steps
    assert np.all([datap["other_info"][1] == x for x in datap["other_info"]])
    # action_dist_inputs = datap["pre_batch"][0]["action_dist_inputs"]
    # actions = datap["pre_batch"][0]["actions"]
    # datap["step"][0]["actio "]

    # -- Get common data for exp iteration
    file_id = i
    training_iteration = datap["other_info"][0]["training_iteration"]
    timesteps_total = datap["other_info"][0]["timesteps_total"]

    common_cl = ["file_id", "training_iteration", "timesteps_total"]
    common_data = [file_id, training_iteration, timesteps_total]

    # -- Get data for each env seed / step
    exp_iter_data_cl = ["smiles", "action_dist_inputs", "action_mask", "step_data", "batch"]
    exp_iter_data = []

    for step_data, batch in zip(datap["step"], datap["pre_batch"]):
        all_actions_r.append([step_data["smiles"], step_data["blockidxs"], step_data["action"], step_data["n_reward"],
                              step_data["n_info"]["log_vals"]['proxy_dock'][0],
                              step_data["n_info"]["log_vals"]['qed_score'],
                              step_data["n_info"]["log_vals"]['true_dockscore'],
                              step_data["n_info"]["log_vals"]['synth_score']])
        smi = step_data["smiles"]
        action_dist_inputs = batch["action_dist_inputs"]
        act_mask = env_data[smi]["obs"]["action_mask"].astype(np.bool)
        num_next_act = env_data[smi]["obs"]["action_mask"].sum()

        assert np.all(action_dist_inputs[:, ~act_mask] == -20), "Not same act mask"
        assert np.all(action_dist_inputs[:, act_mask] != -20), "Not same neg act mask"
        # assert len(env_data[smi]["next_state"]) ==
        # dist = action_dist_inputs[:, act_mask]
        # dist = dist / dist.sum()

        # step_act = step_data["action"]
        # proxy_dock = step_data["n_info"]["log_vals"]["proxy_dock"][0]
        # synth_score = step_data["n_info"]["log_vals"]["synth_score"]
        # qed_score = step_data["n_info"]["log_vals"]["qed_score"]
        # dockscore = step_data["n_info"]["log_vals"]["true_dockscore"][0]

        # next_smi = step_data["n_info"]["molecule"]
        # assert env_data[smi]["next_state"][105]["mol_smiles"] == next_smi, "Test chosen next smi"

        exp_iter_data.append([smi, action_dist_inputs, act_mask, step_data, batch])

    df_exp_iter = pd.DataFrame(exp_iter_data, columns=exp_iter_data_cl)
    for xcl, xdata in zip(common_cl, common_data):
        df_exp_iter[xcl] = xdata

    # Add next
    df_exp_iter["next"] = df_exp_iter.smiles.apply(lambda x: env_data[x]["next"])

    exp_data.append(df_exp_iter)
    # assert df_exp_iter.smiles.nunique() == len(df_exp_iter), "What"

df_exp = pd.concat(exp_data)
df_exp_orig = df_exp.copy()

# df_exp = df_exp[df_exp.training_iteration == 280]
df_exp["action_dist_inputs"] = df_exp["action_dist_inputs"].apply(lambda x: x.reshape(-1))
assert (df_exp.groupby("file_id")["training_iteration"].nunique()==1).all(), "1 Unique tr_step/iter"


group = list(df_exp.groupby("timesteps_total"))[0][1] # 1 group
print("MAX possible mean R", group.smiles.apply(lambda x: env_data[x]["next"]["true_r"].max()).mean())

# Calculate top match
# TODO FIX this - order can contain masked actions
df_exp["dist_act_order"] = df_exp.apply(lambda x: np.argsort(x["action_dist_inputs"])[::-1], axis=1)
# df_exp["dist_act_order_g"] = df_exp.apply(
#     lambda x: [x["next"].loc[y]["true_r_group"]
#                if y in env_data[x["smiles"]]["next"].index else np.nan
#                for y in x["dist_act_order"]], axis=1)


tgt_sort = "true_r"
# df_exp["target_act_order"] = df_exp.smiles.apply(lambda x: env_data[x]["next"][env_data[x]["next"]["candidate"]].sort_values(by=tgt_sort)["action"].values)
# df_exp["target_act_order"] = df_exp.smiles.apply(lambda x: env_data[x]["next"].sort_values(by=tgt_sort)["action"].values[::-1])
df_exp["target_r"] = df_exp.next.apply(lambda x: x.sort_values(by=tgt_sort)[tgt_sort].values[::-1])
df_exp["target_act_order"] = df_exp.next.apply(lambda x: x.sort_values(by=tgt_sort).index.values[::-1])
df_exp["target_act_order_g"] = df_exp.next.apply(lambda x: x.sort_values(by=tgt_sort)["true_r_group"].values[::-1])
# df_exp["target_act_order_cand"] = df_exp.smiles.apply(lambda x: env_data[x]["next"][env_data[x]["next"]["candidate"]].sort_values(by=tgt_sort).index.values[::-1])

df_exp["tgt_act_dist"] = df_exp.apply(lambda x: x["action_dist_inputs"][x["target_act_order"]], axis=1)
df_exp["pred_act_dist"] = df_exp.apply(lambda x: x["action_dist_inputs"][x["dist_act_order"]], axis=1)

# ==================================================================================================

for irow, row in df_exp.iterrows():
    nxt = env_data[row["smiles"]]["next"]
    dist_order = row["dist_act_order"]
    row["action_dist_inputs"]
    order = nxt.loc[dist_order]
df_exp["pred_act_r"] = df_exp.apply(lambda x: env_data[x["smiles"]]["next"].loc[].values, axis=1)
# ==================================================================================================
# Check good seeds
for k, v in env_data.items():
    print(v["next"]["candidate"].sum(), k, v["env_seed"])

print(env_data["O=CNc1cncc(C2CCCC2)c1"]['next']["true_r"].max())
print(env_data["CC(C)C(NC=O)C(C)(C)O"]['next']["true_r"].max())
print(env_data["FC1CCCC(c2nccs2)C1"]['next']["true_r"].max())
env_data["O=CNc1cncc(C2CCCC2)c1"]['next']["true_r"].max() + env_data["CC(C)C(NC=O)C(C)(C)O"]['next']["true_r"].max() + env_data["FC1CCCC(c2nccs2)C1"]['next']["true_r"].max()
# ==================================================================================================
# Check why true_r different from qued
all_data_r = pd.DataFrame(all_actions_r, columns=["smiles", "blockidxs", "action", "n_reward",
                                                  "proxy_dock", "qed_score", "true_dockscore",
                                                  "synth_score"])
wrong_is = []
for i, row in all_data_r.iterrows():
    # break
    next_d = env_data[row["smiles"]]["next"]
    selection = next_d.loc[row["action"]]
    truer_env_data = selection["true_r"]
    r_env = row["n_reward"]
    if truer_env_data != r_env:
        wrong_is.append([i, truer_env_data, r_env, selection])

len(all_data_r)
len(wrong_is)
i = 3
cnt = 0
difs = []
for i in range(len(wrong_is)):
    difs.append(wrong_is[i][1] - wrong_is[i][2])
    if (wrong_is[i][1] - wrong_is[i][2]) > 0.00001:
        cnt+=1
        print(wrong_is[i][1:3])
        print("exp_data", all_data_r.loc[wrong_is[i][0]]["true_dockscore"])
        print("env_data", wrong_is[i][3][["dockscore"]])
# ==================================================================================================

# df_exp["dist_act_order"].apply(lambda x: set(x[:top_k]))
top_k = 1


for top_k in [1,3,5]: #,10,15]:
    df_exp["dist_act_order_g"] = df_exp.apply(
        lambda x:
        env_data[x["smiles"]]["next"].loc[
            x["dist_act_order"][:top_k][x["action_mask"][x["dist_act_order"][:top_k]]]][
            "true_r_group"].values, axis=1)

    tgt_cl = "target_act_order_g" # target_act_order
    # df_exp["rank_match"] = df_exp.apply(
    #     lambda row: (len(set.intersection(
    #         set(row[tgt_cl][:top_k]), set(row["dist_act_order_g"][:top_k]))) +EPS) / float(len(set(row[tgt_cl][:top_k]))+EPS), axis=1)

    df_exp["top_k_tgt"] = df_exp.apply(lambda x: set(env_data[x["smiles"]]["target_r_groups"][:top_k]), axis=1)
    df_exp["top_k_pred"] = df_exp.dist_act_order_g.apply(lambda x: set(x))
    df_exp["rank_match"] = df_exp.apply(
        lambda row: (len(set.intersection(row["top_k_pred"], row["top_k_tgt"])) +EPS) /
                    float(len(row["top_k_pred"])+EPS), axis=1)
    #
    plt_values = df_exp.groupby("timesteps_total")["rank_match"].mean()
    plt_values.plot()
    plt_exps.append([exp_path, top_k, deepcopy(plt_values)])

# ==================================================================================================
# plt_exps = []

ranks = pd.DataFrame(plt_exps, columns=["path", "top_k", "series"])
name_change = dict({"rlbo_eval_50_PPO": "eval_50m_Default",
                    "rlbo_eval_test_50": "eval_50m_MLP",
                    "rlbo_eval_debug_50": "eval_50m_MPNNet_v2",
                    })
ranks["name"] = ranks.path.apply(lambda x: name_change[os.path.basename(x)])

for r in ranks.top_k.unique():
    rf = ranks[ranks.top_k == r]

    fig, ax = plt.subplots()
    for _id, row in rf.iterrows():
        row["series"].plot(ax=ax, label=row["name"])
    plt.legend(loc="lower right")
    ax.set_title(f"Fraction correct Top-{r} act predictions")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("fraction")
    plt.savefig(f"data/plots/top_{r}.png")
# ==================================================================================================
import torch
from torch.distributions import Categorical

test_ = torch.ones_like(torch.from_numpy(df_exp.iloc[0]["action_dist_inputs"]))
test_[:] = -10
test_[0] = 0
print(Categorical(logits=test_).entropy())

df_exp["entropy"] = df_exp.action_dist_inputs.apply(lambda x: Categorical(logits=torch.from_numpy(x)).entropy().item())
plt_values = df_exp.groupby("timesteps_total")["entropy"].mean()
plt_values.plot()

# ==================================================================================================
# check order at end of training based on rank
max_blocks = 7
num_blocks = 105
df_exp["action"] = df_exp['step_data'].apply(lambda x: x["action"])
df_exp["true_r"] = df_exp.apply(lambda x: env_data[x["smiles"]]["next"].loc[x["action"]]["true_r"], axis=1)
df_exp["blockidxs"] = df_exp['step_data'].apply(lambda x: x["blockidxs"])
df_exp["jbonds"] = df_exp['step_data'].apply(lambda x: x["jbonds"])
df_exp["stems"] = df_exp['step_data'].apply(lambda x: x["stems"])

df_exp["n_reward"] = df_exp['step_data'].apply(lambda x: x["n_reward"])
df_exp["stem_idx"] = df_exp["dist_act_order"].apply(lambda x: [(y - max_blocks) // num_blocks for y in x])
df_exp["block_idx"] = df_exp["dist_act_order"].apply(lambda x: [(y - max_blocks) % num_blocks for y in x])

last_iter = df_exp[df_exp.training_iteration == df_exp.training_iteration.max()]
print(last_iter["block_idx"].value_counts())
print(last_iter.rank_match.value_counts())

smiles_by_rank = last_iter.groupby("rank_match").smiles.unique()
smiles_by_rank = smiles_by_rank.sort_index()
print(smiles_by_rank.apply(len))


df_exp["t_stem_idx"] = df_exp["target_act_order"].apply(
    lambda x: [(y - max_blocks) // num_blocks for y in x])
df_exp["t_block_idx"] = df_exp["target_act_order"].apply(
    lambda x: [(y - max_blocks) % num_blocks for y in x])

for smi_rank, group in smiles_by_rank.iteritems():
    env_data[group[0]]["next"]["true_r"].describe()
    print("SMI_RANK", smi_rank, group)

    group_idx = 0
    flts = []
    for group_idx in range(len(group)):
        flt = df_exp[df_exp.smiles == group[group_idx]][["dist_act_order", "target_act_order", "training_iteration", "action", "stem_idx", "block_idx","n_reward", "blockidxs", "stems", "smiles", "target_r"]].copy()
        flt = flt.sort_values("training_iteration")

        flts.append(flt.tail(1))
    flts = pd.concat(flts)
    flts.to_clipboard()
    break

    print(flts)
    input()
# ==================================================================================================
vs = []
for line in lines:
    if "[48, 90, 101]" in line and len(line)>26:
        vs.append([int(line.strip().split()[-2]), float(line.strip().split()[-1])])
dfq = pd.DataFrame(vs,columns=["action", "reward"])

dfq["envr"] = dfq.action.apply(lambda x: env_data[df_exp.iloc[4]["smiles"]]["next"].loc[x]["true_r"])
same = []
for i, rrr in dfq.iterrows():
    same.append([, rrr["reward"]])
# ==================================================================================================
# Look Individually at each test env
fig = None
for ip, (label, df_group) in enumerate(df_exp.groupby("smiles")):
    if ip % 10 == 0:
        if fig is not None:
            plt.show(block=False)
            plt.pause(0.001)  # Pause for interval seconds.
            input("hit[enter] to end.")
        plt.close('all')

        fig, ax = plt.subplots()
    df_group.sort_values("timesteps_total").plot(y="rank_match", x="timesteps_total", ax=ax, label=label)

# ==================================================================================================
# Check R scores
all_next_stat_r = df_exp.smiles.unique().apply(lambda x: env_data[x]["next"]["true_r"])
next_stat_num_act = df_exp.smiles.apply(lambda x: len(env_data[x]["next"]))
print("Same actions as r!=nan:", ((~all_next_stat_r.isna()).sum(axis=1) == next_stat_num_act).all())

next_stat_r = df_exp.groupby("smiles").head(1).smiles.apply(lambda x: env_data[x]["next"]["true_r"])
next_stat_cand = df_exp.groupby("smiles").head(1).smiles.apply(lambda x: env_data[x]["next"]["candidate"])

qed_cutoff = [0.2, 0.5]
synth_cutoff = [0, 4]
min_clip = df_exp.groupby("smiles").head(1).smiles.apply(
    lambda x: (env_data[x]["next"]["qed_score"] > qed_cutoff[0]) &
              (env_data[x]["next"]["synth_score"] > synth_cutoff[0]))

nonzero_r = (((~next_stat_r.isna()) & (next_stat_r != 0)).sum(axis=1) / next_stat_num_act)

# plot types
pltdf = pd.DataFrame()
pltdf["num_act"] = (~next_stat_r.isna()).sum(axis=1)
pltdf["nonzero_r"] = ((~next_stat_r.isna()) & (next_stat_r != 0)).sum(axis=1)
pltdf["cand"] = next_stat_cand.sum(axis=1)
pltdf["min_clip"] = min_clip.sum(axis=1)
pltdf.plot(kind="bar")  # Number of nonzero R

# Lots of action candidate
dft = df_exp.groupby("smiles").head(1).iloc[39]
smi = dft.smiles

print("Num valid act", dft.action_mask.sum())

ax = env_data[smi]["next"][["qed_score", "synth_score", "dockscore", "true_r"]].plot(subplots=True, style='.-')

env_data[smi]["next"]["true_r"]

# ==================================================================================================
# # ======================================================
# # test

idx = 10
drd = []
for idx in range(len(df_exp)):
    smi = df_exp.iloc[idx].smiles

    # print(df_exp[["dist_act_order", "target_act_order"]])
    # print(db.loc[smi]["dockscore"])
    stpe_act = df_exp.iloc[idx]["step_data"]["action"]
    expr = df_exp.iloc[idx]["step_data"]["n_reward"]
    dbr = env_data[smi]["next"][env_data[smi]["next"]["action"]==stpe_act]["true_r"].values[0]
    drd.append([smi, stpe_act, expr, dbr])
drd = pd.DataFrame(drd, columns=["smi", "stpe_act", "expr", "dbr"])
print("N r", expr, )
print()

# env_data[smi]["next"]["true_r"]
# env_data[smi]["next"]["dockscore"]
# ==================================================================================================
