import glob
import time
from select import epoll

import gym
import numpy as np
import time

import pandas as pd

from train_light import preprocess_args, set_seed, setup_loggers, get_envs, pre_process_obss
from lightrl.env.vec_env import fast_from_data_list
from torch_geometric.data import Batch

import torch


def fast_from_data_list(data_list, inckeys=set(['stem_atmidx','edge_index','jbond_atmidx'])):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly.
    """

    keys = data_list[0].keys

    batch = Batch()
    batch.__data_class__ = data_list[0].__class__
    batch.__slices__ = {key: [0] for key in keys}

    for key in keys:
        batch[key] = []

    cumsum = [0] * len(keys)
    batch.batch = []
    for j, key in enumerate(keys):
        cat_dim = 1 if key == 'edge_index' else 0
        slc = batch.__slices__[key]
        bk = batch[key]
        for i, data in enumerate(data_list):
            item = data[key]
            if cumsum[j] > 0:
                item = item + cumsum[j]
            slc.append(item.shape[cat_dim] + slc[-1])
            if key in inckeys:
                cumsum[j] += data.x.shape[0]
            bk.append(item)
            if j == 0:
                batch.batch.append(torch.full((data.x.shape[0], ), i, dtype=torch.long))

    for key in batch.keys:
        item = batch[key][0]
        cd = data_list[0].__cat_dim__(key, item)
        batch[key] = torch.cat(batch[key], dim=cd)

    return batch.contiguous()


def gen_obs_file():
    from lightrl import env
    import torch
    import gym
    import copy
    from LambdaZero.contrib.config_eval_env import ALL_SEEDS, SEED_FILTER

    seed_cnt = 500

    env = gym.make("BlockMolEnvGraph-v1", config={
        "random_steps": 3,
        "max_steps": 1,
        "allow_removal": True,
        "env_seed": ALL_SEEDS[:seed_cnt].tolist()},
        proc_id=0
    )

    obs_mol_graph = dict()
    obs_mols = dict()

    def add_smi(_smi, _graph, check=False, other=None):
        if _smi in obs_mol_graph:
            mem_graph = obs_mol_graph[_smi]
            if check:
                for k in _graph.keys:
                    if not torch.all(_graph[k] == mem_graph[k]):
                        import pdb; pdb.set_trace()

                    assert torch.all(_graph[k] == mem_graph[k]), f"Must be the same {_smi}"
        else:
            obs_mol_graph[_smi] = copy.deepcopy(_graph)
            # obs_mols[_smi] = copy.deepcopy(other)

    for i_seed in range(seed_cnt):
        print(f"Doing {i_seed}/{seed_cnt}")
        obs = env.reset(episode=i_seed)
        smi = env.molMDP.molecule.smiles
        add_smi(smi, obs["mol_graph"], other={"mol": env.molMDP.molecule})

        action_mask = obs["action_mask"]
        valid_actions = env._get_av_actions(action_mask)

        for act in valid_actions:
            obs = env.reset(episode=i_seed)
            obs, reward, done, info = env.step(act)
            add_smi(env.molMDP.molecule.smiles, obs["mol_graph"], other={"mol": env.molMDP.molecule})

    torch.save(obs_mol_graph, "data/mol_data.pkl")


def test_env():
    from lightrl import env
    import gym
    from lightrl.env.vec_env import get_envs, pre_process_obss

    env = gym.make("BlockMolEnvGraph-v1")

    obs = env.reset()

    # obs.pop("mol_graph")
    device = torch.device("cuda")
    st = time.time()
    for _ in range(500):
        obss = pre_process_obss([obs] * 256, device=device)
    print(time.time() - st)

    obss2 = Batch.from_data_list([mol_graph, mol_graph, mol_graph])

class BrutalDock(InMemoryDataset):
    # own internal dataset
    def __init__(self, data, slices, root=None, transform=None, pre_transform=None, pre_filter=None):
        root ="/media/andrei/CE04D7C504D7AF292/milaintern/LambdaZero/dataset/brutal_dock/seh/"

        super(BrutalDock, self).__init__(root, transform, pre_transform, pre_filter)

        processed_paths = "/media/andrei/CE04D7C504D7AF292/milaintern/lz_branch/data/large_random.pt"
        graphs = []
        for processed_path in [processed_paths]:
            self.data, self.slices = torch.load(processed_path)
            graphs += [self.get(i) for i in range(len(self))]

        if len(graphs) > 0:
            self.data, self.slices = self.collate(graphs)

    @property
    def raw_file_names(self):
        return ["/media/andrei/CE04D7C504D7AF292/milaintern/lz_branch/data/large_random.feather"]

    @property
    def processed_file_names(self):
        return ["/media/andrei/CE04D7C504D7AF292/milaintern/lz_branch/data/large_random.pt"]

    def download(self):
        pass

def compare():
    import torch
    import yaml
    from liftoff import dict_to_namespace
    from train_reg_light import ListGraphDBDataset, load_predocked_data
    import numpy as np
    from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
    import LambdaZero

    transform = LambdaZero.utils.Complete()

    saved_data = BrutalDock(None, None, transform=transform)

    with open("lightrl/configs/regressor.yaml") as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    cfg = dict_to_namespace(config_data)

    graphs = load_predocked_data(cfg.dataset)
    dataset = ListGraphDBDataset(graphs, cfg.dataset)

    idx = np.random.choice(len(dataset))
    graph = dataset[idx]
    smiles = graph.smiles

    datap_idx = saved_data.data.smiles.index(smiles)
    saved_graph = saved_data[datap_idx]

    (graph.x == saved_graph.x).all(dim=0)
    (graph.edge_index == saved_graph.edge_index).all()

def run(args):
    args = preprocess_args(args)

    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    setup_loggers(args)

    # torch.set_num_threads(1)  # TODO why?
    device = args.device

    send_reset_info = True
    eval_eps = 1000
    envs, chunk_size = get_envs(args.env_cfg, args.env_cfg.procs, send_reset_info=True)

    obss = envs.reset(num_evals=eval_eps)

    obss = pre_process_obss(obss)

    # ==============================================================================================
    # test speed
    num_steps = 100
    obss = envs.reset()
    st = time.time()
    for ist in range(num_steps):
        action_masks = [obs["action_mask"] for obs in obss]
        valid_actions = [envs.first_env._get_av_actions(action_mask) for action_mask in
                         action_masks]
        act = [np.random.choice(x) for x in valid_actions]
        obs, reward, done, info = envs.step(act)
        import pdb; pdb.set_trace()

    print(f"Run {num_steps * args.env_cfg.procs / (time.time() - st)}")

    exit()

def test1():
    import torch
    import glob
    import matplotlib
    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np

    dps = "train_data_1"
    data_p = f"/media/andrei/CE04D7C504D7AF292/milaintern/lz_branch/data/debug_r/{dps}"
    fl_pths = glob.glob(f"{data_p}/*")
    datas = []
    for pth_id in range(len(fl_pths)):
        datas.append(torch.load(f"{data_p}/update_{pth_id}"))

    info = [save_data["info"] for save_data in datas]
    act = [save_data["act"] for save_data in datas]
    r = [save_data["r"] for save_data in datas]
    done = [save_data["done"] for save_data in datas]
    ep_r1 = [save_data["ep_r"] for save_data in datas]
    ep_r2 = [save_data["ep_r"] for save_data in datas]

    dfs = []
    for update in range(len(info)):
        for step in range(len(info[update])):
            for x_i, x_a in zip(info[update][step], act[update][step]):
                x_i["act"] = x_a

            df = pd.DataFrame.from_dict(info[update][step])
            df['update'] = update
            df["step"] = step
            dfs.append(df)
        print(f"done update {update}")

    df = pd.concat(dfs)
    df.to_pickle(f"data/debug_r/{dps}.pkl")

    # ==
    df1 = pd.read_pickle(f"data/debug_r/train_data_1.pkl")
    df2 = pd.read_pickle(f"data/debug_r/train_data_2.pkl")
    df1["env_id"] = df1.index
    df2["env_id"] = df2.index
    df1 = df1.sort_values(["update", "step", "env_id"]).reset_index()
    df2 = df2.sort_values(["update", "step", "env_id"]).reset_index()

    df1["reward"] = df1["log_vals"].apply(lambda x: x["reward"] if isinstance(x, dict) else -1)
    df2["reward"] = df2["log_vals"].apply(lambda x: x["reward"] if isinstance(x, dict) else -1)

    dfs = df[df["update"] > 150]
    print(dfs.groupby("act_molecule")["reward"].nunique())
    print([np.mean(x) for x in ep_r])

    ax = df1.groupby("update")["reward"].mean().plot()
    df2.groupby("update")["reward"].mean().plot(ax=ax)

    df1.groupby(["update", "act_molecule"])["act"].nunique()

    diff = df1.groupby(["update", "act_molecule"])["act"].nunique().loc[185] - df1.groupby(["update", "act_molecule"])["act"].nunique().loc[0]
    diff = pd.DataFrame(diff)
    diff["nunique"] = df1.groupby(["update", "act_molecule"])["act"].nunique().loc[185]

    r_g = df1.groupby("update").tail(100)



    plt.plot([np.mean(x) for x in ep_r1])
    plt.plot([np.mean(x) for x in ep_r2])
    plt.show()

    same = df1[df1.res_molecule == df2.res_molecule]
    same.groupby("update")["index"].count().plot()
    same2 = df1[df1.act == df2.act]
    same2.groupby("update")["index"].count().plot()

def test2():
    from LambdaZero.contrib.config_eval_env import ALL_SEEDS
    import LambdaZero
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    seeds = ALL_SEEDS[:]

    datasets_dir = "data/"

    data = np.load(f"{datasets_dir}/eval_env_data.npy", allow_pickle=True).item()  # type: dict

    nexts = []
    for smi, smi_data in data.items():
        if smi_data["env_seed"][0] in seeds:
            smi_data["next"]["smi"] = smi
            nexts.append(smi_data["next"])

    df = pd.concat(nexts)
    df = df.reset_index()

    # Top 1% act
    top_act_p = 0.01
    top_act = df.sort_values("true_r", ascending=False).head(int(len(df) * top_act_p))

    top_mol_cnt = top_act.groupby("smi").action.count().sort_values()

    plt.plot(list(range(len(top_mol_cnt))), top_mol_cnt.values)
    plt.show()
    top_mol_cnt.plot(rot=0.46)
    num_seeds = len(seeds)


    scores = []
    batch = 100
    rnd_state = np.random.RandomState(10)
    for i in range(100):
        ep_scores = []
        for b in range(batch):
            r = nexts[rnd_state.randint(num_seeds)]["true_r"].sample(1).values[0]
            ep_scores.append(r)
        scores.append(np.mean(ep_scores))

    scores1 = scores
    scores2 = scores

    plt.plot(scores1)
    plt.plot(scores2)
    plt.show()



if __name__ == "__main__":
    gen_obs_file()