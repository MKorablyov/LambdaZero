import itertools
import sys
import os.path as osp
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score

import torch
from torch_geometric.data import DataLoader
from argparse import Namespace
import numpy as np
from sklearn.metrics import explained_variance_score
from copy import deepcopy
import torch.nn.functional as F
import wandb
import os
from torch.utils.data import Dataset
import gym
from lightrl import env
from torch_geometric.utils import degree

from lightrl.utils.utils import set_seed
from lightrl.env.vec_env import fast_from_data_list
from lightrl.utils.utils import parse_opts, add_to_cfg, flatten_cfg, update_cfg

import LambdaZero.models
from LambdaZero.examples.mpnn import config
from LambdaZero.utils import get_external_dirs
from LambdaZero.contrib.inputs import ListGraphDataset
from LambdaZero.contrib.inputs import temp_load_data_v1
from LambdaZero.contrib.functional import elu2
from LambdaZero.environments.molMDP import BlockMoleculeData

from lightrl.reg_models import get_reg_model
from lightrl.utils.radam_optimizer import RAdam

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

DOCK_MEAN = -8.6
DOCK_STD = 1.1

load_seen_config = {
    "mean": -8.6,
    "std": 1.1,
    "act_y": elu2,
    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
    "raw_path": osp.join(datasets_dir, "brutal_dock/seh/raw"),
    "proc_path": osp.join(datasets_dir, "brutal_dock/seh/processed_rlbo"),
    "file_names": ["Zinc20_docked_neg_randperm_30k"],
}

LOG_HEADER = {
}


def train_epoch(loader, model, optimizer, device, max_epoch=np.inf, batch_gradient=1):
    model.train()
    train_mse_loss = []
    train_explained_variance = []

    assert batch_gradient == 1, f"Not implemented for {batch_gradient}"

    cnt_data = 0
    last_log_cnt = 0

    normalizer = model._norm

    st = time.time()
    for bidx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        y_hat = model(data)
        tgt = normalizer.tfm(data.dockscore)
        loss = F.mse_loss(y_hat, tgt)
        if getattr(model, "has_loss", False):
            loss += model._loss

        loss.backward()
        optimizer.step()

        epoch_y = tgt.detach().cpu().numpy()
        epoch_y_hat = y_hat.detach().cpu().numpy()
        train_mse_loss.append(loss.item())
        train_explained_variance.append(explained_variance_score(epoch_y, epoch_y_hat))

        cnt_data += len(tgt)

        if (cnt_data - last_log_cnt) > 1000:
            print(f"[{bidx}] loss {np.mean(train_mse_loss)}")
            last_log_cnt = cnt_data

        if cnt_data > max_epoch:
            break

    end_time = time.time()-st
    print(f"Train time: {end_time} (train time / 1000mol {(end_time/ cnt_data) * 1000})")

    return {"model/train_loss": np.mean(train_mse_loss),
            "model/train_explained_variance": np.mean(train_explained_variance)
            }


def train_class_epoch(loader, model, optimizer, device, max_epoch=np.inf, batch_gradient=1):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    train_mse_loss = []

    cnt_data = 0
    correct = 0
    last_log_cnt = 0

    st = time.time()

    optimizer.zero_grad()

    for bidx, data in enumerate(loader):
        data = data.to(device)

        if batch_gradient == 1:
            optimizer.zero_grad()
            y_hat = model(data)
            loss = criterion(y_hat, data.label)
            if getattr(model, "has_loss", False):
                loss += model._loss

            loss.backward()
            optimizer.step()
        elif (bidx + 1) % batch_gradient == 0:
            y_hat = model(data)
            loss = criterion(y_hat, data.label)
            if getattr(model, "has_loss", False):
                loss += model._loss

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        else:
            y_hat = model(data)
            loss = criterion(y_hat, data.label)
            if getattr(model, "has_loss", False):
                loss += model._loss

            loss.backward(retain_graph=True)

        train_mse_loss.append(loss.item())

        _, predicted = torch.max(y_hat.data, 1)
        correct += (predicted == data.label).sum().item()

        cnt_data += len(data.label)

        if (cnt_data - last_log_cnt) > 1000:
            print(f"[{bidx}] loss {np.mean(train_mse_loss)}")
            last_log_cnt = cnt_data

        if cnt_data > max_epoch:
            break

    end_time = time.time()-st
    print(f"Train time: {end_time} (train time / 1000mol {(end_time/ cnt_data) * 1000})")

    return {"model/train_loss": np.mean(train_mse_loss),
            "model/train_acc": 100 * correct / cnt_data
            }

def construct_metrics(epoch_targets, epoch_preds, epoch_d_targets, epoch_d_preds, prefix=""):
    metrics = dict()
    metrics[f"{prefix}_loss"] = ((epoch_targets-epoch_preds)**2).mean()
    metrics[f"{prefix}_mae"] = np.abs(epoch_d_targets - epoch_d_preds).mean()
    metrics[f"{prefix}_mse"] = ((epoch_d_targets - epoch_d_preds) ** 2).mean()
    metrics[f"{prefix}_explained_variance"] = explained_variance_score(epoch_targets, epoch_preds)
    return metrics

def val_epoch(loader, model, device):
    model.eval()
    target_y = []
    target_dock = []
    pred_y = []
    bins = []
    normalizer = model._norm

    with torch.no_grad():
        for bidx, data in enumerate(loader):
            data = data.to(device)
            y_hat = model(data)
            tgt = normalizer.tfm(data.dockscore)

            target_y.append(tgt.detach().cpu().numpy())

            target_dock.append(data.dockscore.detach().cpu().numpy())
            pred_y.append(y_hat.detach().cpu().numpy())
            # bins.append(data.bin.detach().cpu().numpy())

    target_y = np.concatenate(target_y, 0)
    target_dock = np.concatenate(target_dock, 0)
    pred_y = np.concatenate(pred_y, 0)
    pred_dock = model._norm.itfm(pred_y)
    # bins = np.concatenate(bins, 0)

    scores = construct_metrics(target_y, pred_y, target_dock, pred_dock, prefix="model/val")

    bin_cnts = dict()
    # for ibin in np.unique(bins):
    #     flt = bins == ibin
    #     bin_cnts[ibin] = bin_cnt = flt.sum()
    #     if bin_cnt == 0:
    #         continue
    #     bin_target_y, bin_pred_y, bin_target_dock, bin_pred_dock = \
    #         target_y[flt], pred_y[flt], target_dock[flt], pred_dock[flt]
    #
    #     bscores = construct_metrics(bin_target_y, bin_pred_y, bin_target_dock, bin_pred_dock,
    #                                 prefix=f"model/val_bin_{ibin}")
    #
    #     scores.update(bscores)

    return scores, bin_cnts


def val_class_epoch(loader, model, device):
    model.eval()
    epoch_y = []
    epoch_y_hat = []
    epoch_predicted = []
    cnt_data = 0
    correct = 0

    with torch.no_grad():
        for bidx, data in enumerate(loader):
            data = data.to(device)
            y_hat = model(data)

            cnt_data += len(data.label)
            epoch_y.append(data.label.detach().cpu())
            epoch_y_hat.append(y_hat.detach().cpu())

            _, predicted = torch.max(y_hat.data, 1)
            epoch_predicted.append(predicted.detach().cpu())
            correct += (predicted == data.label).sum().item()

    epoch_y = torch.cat(epoch_y)
    epoch_y_hat = torch.cat(epoch_y_hat)
    epoch_predicted = torch.cat(epoch_predicted)

    loss = torch.nn.CrossEntropyLoss()(epoch_y_hat, epoch_y).item()

    prec = precision_score(epoch_y.numpy(), epoch_predicted.numpy())
    recall = recall_score(epoch_y.numpy(), epoch_predicted.numpy())

    scores = {
        "model/val_loss": loss,
        "model/val_acc": 100 * correct / cnt_data,
        "model/val_prec": 100 * prec,
        "model/val_recall": 100 * recall,
    }

    return scores, {}


def get_mol(state, molMDP):
    mol = BlockMoleculeData()
    mol.blockidxs = state["blockidxs"]  # indexes of every block
    mol.slices = state["slices"]  # atom index at which every block starts
    mol.jbonds = state["jbonds"]  # [block1, block2, bond1, bond2]
    mol.stems = state["stems"]  # [block1, bond1]
    mol.numblocks = len(mol.blockidxs)
    mol.blocks = [molMDP.block_mols[idx] for idx in mol.blockidxs]  # rdkit molecule objects for every
    mol._mol = None
    return mol


def pool_get_mol_graph(largs):
    state, env_molMDP, env_graph_mol_obs, transform = largs
    mol = get_mol(state, env_molMDP)
    graph, flat_graph = env_graph_mol_obs(mol, flatten=False)
    if transform is not None:
        graph = transform(graph)
    return graph


import LambdaZero
from rdkit import Chem

from rdkit.Chem import AllChem
from LambdaZero.inputs.inputs_op import mpnn_feat, _mol_to_graph


def build_mol(smiles=None, num_conf=1, minimize=False, noh=True, charges=True):
    # todo: things other than SMILES
    # fixme: return numpy array and not list
    # todo: select min energy?
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    # create and optimize 3D structure
    if num_conf > 0:
        assert not "h" in set([atm.GetSymbol().lower() for atm in mol.GetAtoms()]), "can't optimize molecule with h"
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf)
        if minimize:
            [AllChem.MMFFOptimizeMolecule(mol, confId=i) for i in range(num_conf)]
        if charges:
            AllChem.ComputeGasteigerCharges(mol)
            #print("charge 0",float(mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')))
        if noh:
            mol = Chem.RemoveHs(mol)
    return mol


def mol_to_graph(smiles, props={}, num_conf=1, noh=True, feat="mpnn"):
    """mol to graph convertor"""
    mol = build_mol(smiles, num_conf=num_conf, noh=noh)
    if feat == "mpnn":
        atmfeat, coord, bond, bondfeat = mpnn_feat(mol, ifcoord=False)
    else:
        raise NotImplementedError(feat)
    graph = _mol_to_graph(atmfeat, coord, bond, bondfeat, props)
    return graph


def pool_get_mol_graph2(largs):
    state, env_molMDP, env_graph_mol_obs, transform = largs

    graph = mol_to_graph(state["smiles"], ["dockscore"])

    if transform is not None:
        graph = transform(graph)
    return graph


def load_predocked_dataset(args: Namespace):
    import pandas as pd
    import gym
    from lightrl import env
    import json
    import pandas as pd
    import numpy as np
    from multiprocessing import Pool
    import functools
    from LambdaZero.contrib.oracle.oracle import PreDockingDB

    num_pool = getattr(args, "mol_graph_pool", 10)
    max_mol = getattr(args, "max_mol", None)

    pathdb, all_dbs = PreDockingDB.get_last_db()
    columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
    store = pd.HDFStore(pathdb, "r")
    df = store.select('df')

    if max_mol is not None and max_mol != 0:
        df = df.iloc[:max_mol]

    for cl_mame in columns[2:]:
        df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)
    df.dockscore = df.dockscore.apply(lambda x: np.around(np.float64(x), 1))

    env = gym.make("BlockMolEnvGraph-v1", config={
        "random_steps": 3,
        "max_steps": 1,
        "allow_removal": True,
        "env_seed": 0}, proc_id=0)

    if args.transform_complete:
        transform = LambdaZero.utils.Complete()
    else:
        transform = None

    with Pool(num_pool) as p:
        graphs = p.map(
            pool_get_mol_graph,
            zip(df.to_dict(orient='records'),
                itertools.repeat(env.molMDP),
                itertools.repeat(env.graph_mol_obs),
                itertools.repeat(transform))
        )
    for g, y in zip(graphs, df.dockscore.values):
        g.y = torch.FloatTensor([y])
    return graphs


def load_predocked_data(args: Namespace):
    import pandas as pd
    import json
    import pandas as pd
    import numpy as np
    from multiprocessing import Pool
    import functools
    from LambdaZero.contrib.oracle.oracle import PreDockingDB

    num_pool = getattr(args, "mol_graph_pool", 10)
    max_mol = getattr(args, "max_mol", None)

    pathdb, all_dbs = PreDockingDB.get_last_db()

    columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
    store = pd.HDFStore(pathdb, "r")
    df = store.select('df')

    print(f"Before unique count: {len(df)}")
    # Filter unique smiles # TODO should find a better way to take advantage of diff dockscores
    df = df.groupby(df.index).head(1)
    df = df.reset_index()
    print(f"After unique count: {len(df)}")

    if max_mol is not None and max_mol != 0:
        df = df.iloc[:max_mol]

    for cl_mame in columns[2:]:
        df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)
    df.dockscore = df.dockscore.apply(lambda x: np.around(np.float64(x), 1))

    if args.clip_stds is not None:
        stds_min, stds_max = args.clip_stds
        d_mean, d_std = args.target_norm
        df.dockscore = df.dockscore.clip(d_mean - stds_min * d_std, d_mean + stds_max * d_std)

    # Build equal 100 quantiles
    num_bins = 100
    qcut = pd.qcut(df.dockscore.rank(method="first"), num_bins, labels=list(range(num_bins)))
    df["q100"] = qcut

    # Build two classes
    class_th = args.positive_quantile * 100
    assert (class_th % 1 == 0) and (1 < class_th < 99), "Must be % 0.01 & 0.01 < x < 0.99"
    qbins = qcut.values.to_numpy()
    df["label"] = 0
    df["label"][qbins < class_th] = 1

    # Insure classes are separated by a dockscore value
    dfl = df[df["label"] == 1]
    max_dock = dfl.dockscore.max()

    df[df.dockscore <= max_dock]["label"] = 1

    # TODO filter by unique smiles for split
    return df


def split_bins(df, num_qbins=100, eval_quant=[0.01, 0.09, 0.20]):
    qcut = pd.qcut(df.dockscore.rank(method="first"), num_qbins, labels=list(range(num_qbins)))
    df["q100"] = qcut

    cbins = np.array(eval_quant + [1 - sum(eval_quant) * 2] + eval_quant[::-1]) * 100
    assert ((cbins % 1) == 0).all(), f"Bins must be % with 0.01 (1% factor) not ok: {cbins}"
    cbin_edge = np.concatenate([np.zeros((1,)), cbins.cumsum()])

    df["bin"] = 0
    qbins = df["q100"].values.to_numpy()
    for ibin, (i_s, i_f) in enumerate(zip(cbin_edge[:-1], cbin_edge[1:])):
        df.loc[(qbins >= i_s) & (qbins < i_f), "bin"] = ibin

    print(df.groupby("bin").dockscore.describe())

    return df


def get_splits(graphs, split_probs, split_seed, return_ids_only=True):
    rnd_state = np.random.RandomState(split_seed)
    graph_ids = rnd_state.permutation(len(graphs))
    splits = [int(np.round(len(graphs) * x)) for x in split_probs[:-1]]
    splits = np.cumsum([0] + splits + [len(graphs) - np.sum(splits)])
    g_split = list()

    for split_i, split_j in zip(splits[:-1], splits[1:]):
        if return_ids_only:
            g_split.append(graph_ids[split_i: split_j])
        else:
            g_split.append([graphs[x] for x in graph_ids[split_i: split_j]])
    return g_split


def setup_loggers(args: Namespace):
    import wandb
    import csv

    # Load loggers and setup results path
    use_wandb = args.use_wandb

    if use_wandb:
        experiment_name = f"{args.full_title}_{args.run_id}"
        with open(f"{summaries_dir}/wandb_key") as f:
            api_key = f.readlines()[0].strip()

        os.environ['WANDB_API_KEY'] = api_key

        wandb.init(project=args.project_name, name=experiment_name)
        wandb.config.update(dict(flatten_cfg(args)))

    out_dir = args.out_dir

    # flog = open(out_dir + "/logs.csv", 'w')
    # log_writer = csv.DictWriter(flog, LOG_HEADER.keys())
    # log_writer.writeheader()
    # return log_writer


class MeanVarianceNormalizer:
    def __init__(self, mean_and_std):
        self.mean = mean_and_std[0]
        self.std = mean_and_std[1]

    def tfm(self, x):
        """normalize x"""
        x_norm = (x - self.mean) / self.std
        return x_norm

    def itfm(self, x_norm):
        """denormalize x"""
        x = (x_norm * self.std) + self.mean
        return x

class ListGraphDBDataset(Dataset):
    def __init__(self, db, cfg: Namespace):

        transform = cfg.transform_complete
        bins = cfg.eval_quant
        classification = cfg.classification
        edge_weidht = False
        target_norm = cfg.target_norm

        self._norm = MeanVarianceNormalizer(target_norm)

        self.db = db
        env = gym.make("BlockMolEnvGraph-v1", config={
            "random_steps": 3,
            "max_steps": 1,
            "allow_removal": True,
            "env_seed": 0}, proc_id=0)

        self.env_molMDP = env.molMDP
        self.env_graph_mol_obs = env.graph_mol_obs
        self._graph_from_mol = 0

        if transform:
            self.transform = LambdaZero.utils.Complete()
        else:
            self.transform = None

        self.edge_weidht = edge_weidht

        self.bins = False
        if bins is not None:
            cbins = np.array(bins + [1-sum(bins)*2] + bins[::-1]) * 100
            assert ((cbins % 1) == 0).all(), f"Bins must be % with 0.01 (1% factor) not ok: {cbins}"
            cbin_edge = np.concatenate([np.zeros((1,)), cbins.cumsum()])

            db["bin"] = 0
            qbins = db["q100"].values.to_numpy()
            for ibin, (i_s, i_f) in enumerate(zip(cbin_edge[:-1], cbin_edge[1:])):
                db.loc[(qbins >= i_s) & (qbins < i_f), "bin"] = ibin

            self.bins = True

        self.classification = classification

    def __getitem__(self, idx):
        state = self.db.iloc[idx].to_dict()

        graph = self._get_graph(state)

        if self.classification:
            graph.label = torch.LongTensor([state["label"]])
        else:
            graph.y = torch.FloatTensor([self._norm.tfm(state["dockscore"])])

        graph.docks = torch.FloatTensor([state["dockscore"]])

        if self.bins:
            graph.bin = torch.LongTensor([state["bin"]])

        graph.smiles = state["smiles"]
        return graph

    def _get_graph(self, state):
        transform = self.transform
        env_molMDP = self.env_molMDP
        env_graph_mol_obs = self.env_graph_mol_obs

        # # TODO v1
        if self._graph_from_mol == 1:
            graph = mol_to_graph(state["smiles"])
        else:
            mol = get_mol(state, env_molMDP)
            graph, flat_graph = env_graph_mol_obs(mol, flatten=False)

        if transform is not None:
            graph = transform(graph)

        if self.edge_weidht:

            row, col = graph.edge_index
            graph.edge_weight = 1. / degree(col, graph.num_nodes)[col]  # Norm by in-degree.

        return graph

    def __len__(self):
        return len(self.db)


class BSampler:
    def __init__(self, sampler, datset_size):

        self._sampler = sampler
        self._datset_size = datset_size

    def __iter__(self):
        for i in range(self._datset_size // len(self._sampler)):
            yield list(self._sampler)

    def __len__(self):
        return self._datset_size


def run(cfg: Namespace):
    classification = cfg.optim.classification
    set_seed(cfg.seed, cfg.cuda)
    if classification:
        cfg.project_name += "_class"
        cfg.model.num_out = 2
    else:
        cfg.model.num_out = 1

    setup_loggers(cfg)
    do_plot = cfg.use_wandb
    out_dir = cfg.out_dir

    model = get_reg_model(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    model.to(device)
    if cfg.optim.optim == "RAdam":
        optim_class = RAdam
    else:
        optim_class = getattr(torch.optim, cfg.optim.optim)

    optimizer = optim_class(model.parameters(), **cfg.optim.optim_args.__dict__)

    lr_sheduler_type = getattr(cfg.optim, "lr_sheduler", None)
    lr_sheduler = None
    if lr_sheduler_type is not None:
        lr_sheduler = getattr(torch.optim.lr_scheduler, lr_sheduler_type)
        lr_sheduler = lr_sheduler(optimizer, 'min', patience=10, cooldown=10, factor=0.5)

    # ==============================================================================================
    # Configs
    num_train_epochs = cfg.optim.train_epochs
    batch_size = cfg.optim.batch_size

    # ==============================================================================================
    # Calc obs on batch load
    num_workers = cfg.dataset.num_workers

    from LambdaZero.inputs import BrutalDock
    import LambdaZero
    from torch.utils.data import Subset

    transform = LambdaZero.utils.Complete()

    dataset_config = {
        "root": osp.join(datasets_dir, "systematic_mpnn/seh"),
        "props": ["dockscore", "smiles"],
        "transform": transform,
        "file_names": ["large_random"],
    }

    dataset = BrutalDock(**dataset_config)
    train_idxs, val_idxs, _ = np.load(osp.join(datasets_dir, "systematic_mpnn/seh/large_random_split.npy"), allow_pickle=True)
    print("DSIZE", len(train_idxs), len(val_idxs), len(dataset))

    train_idxs = train_idxs[train_idxs < len(dataset)]
    val_idxs = val_idxs[val_idxs < len(dataset)]
    train_set = Subset(dataset, train_idxs.tolist())
    val_set = Subset(dataset, val_idxs.tolist())
    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.optim.batch_size)
    val_loader = DataLoader(val_set, batch_size=cfg.optim.batch_size)

    # Allocate norm
    _norm = MeanVarianceNormalizer(cfg.dataset.target_norm)
    model._norm = _norm
    print("DSIZE", len(train_set), len(val_set))

    print("Loaded data loaders")

    train_e, val_e = train_epoch, val_epoch

    # ==============================================================================================
    if False:
        calc_deg(train_set)

    # ==============================================================================================

    bin_cnts = None
    epoch_size = cfg.optim.epoch_size
    batch_gradient = cfg.optim.batch_gradient
    eval_freq = cfg.eval_freq
    best_score = -np.inf
    compare_key = "model/val_acc" if classification else "model/val_explained_variance"

    for epoch in range(num_train_epochs):
        train_metrics = train_e(train_loader, model, optimizer, device, max_epoch=epoch_size,
                                batch_gradient=batch_gradient)

        print(train_metrics)

        if lr_sheduler is not None:
            lr_sheduler.step(train_metrics["model/train_loss"])
            if do_plot:
                wandb.log({"lr_sheduler_cooldown": lr_sheduler.cooldown_counter})

        if do_plot:
            wandb.log(train_metrics)

        if eval_freq > 0 and (epoch+1) % eval_freq == 0:
            val_metrics, _bin_cnts = val_e(val_loader, model, device)
            print(val_metrics)
            if do_plot:
                wandb.log(val_metrics)

            if bin_cnts is None:
                bin_cnts = _bin_cnts
                for k, v in bin_cnts.items():
                    wandb.run.summary[str(k)] = v

            if val_metrics[compare_key] > best_score:
                best_score = val_metrics[compare_key]
                print(f"New BEST Score: {best_score} (saving model ...)")
                save_data = dict({
                    "state_dict": model.state_dict(),
                    "model_cfg": cfg.model,
                    "cfg": cfg,
                    "epoch": epoch,
                    "score": best_score
                })
                torch.save(save_data, f"{out_dir}/best_model.pk")

    # update internal copy of the model
    model.eval()
    print("TEST"); sys.stdout.flush(); time.sleep(5); os._exit(0)
    sys.exit()
    exit()



def calc_deg(trainset):
    print("CALCULATING deg")
    # Compute in-degree histogram over training data.
    deg = torch.zeros(1000, dtype=torch.long)
    for i in range(len(trainset)):
        data = trainset[i]

        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    max_deg_idx = torch.where(deg)[0].max().item()
    print(deg[:max_deg_idx])
    exit()

if __name__ == "__main__":
    from lightrl.utils.utils import parse_opts
    run(parse_opts())

#     graphs = load_predocked_dataset(Namespace())
#     print(graphs[0])