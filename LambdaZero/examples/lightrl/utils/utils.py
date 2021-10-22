from argparse import Namespace
import yaml
import os
from typing import List, Dict, Tuple, Any
import torch
import numpy as np
import random
import bisect
import wandb
from collections import deque
from liftoff import OptionParser, dict_to_namespace
import rdkit.DataStructs
from rdkit import Chem
import operator

from LambdaZero.examples.lightrl.env.transforms import \
    TransformInfoDiscounted, TransformCompose, TransformInfoOracle


def flatten_cfg(cfg: Namespace) -> List[Any]:
    """ Flatten nested Namespaces """
    lst = []
    for key, value in cfg.__dict__.items():
        if isinstance(value, Namespace):
            for key2, value2 in flatten_cfg(value):
                lst.append((f"{key}.{key2}", value2))
        else:
            lst.append((key, value))
    return lst


def update_cfg(base_cfg: Namespace, new_cfg: Namespace) -> None:
    """ Inplace update base cfg with new_cfg """
    for k, v in new_cfg.__dict__.items():
        if isinstance(base_cfg.__dict__.get(k, None), Namespace):
            update_cfg(base_cfg.__dict__[k], v)
        else:
            base_cfg.__dict__[k] = v


def add_to_cfg(cfg: Namespace, subgroups: List[str], new_arg: str, new_arg_value) -> None:
    """ Add new_arg to all subgroups items in cfg """
    if subgroups is None:
        subgroups = [x for x in cfg.__dict__.keys() if isinstance(cfg.__dict__[x], Namespace)]

    for arg in subgroups:
        if hasattr(cfg, arg):
            setattr(getattr(cfg, arg), new_arg, new_arg_value)


def parse_opts(check_out_dir: bool = True):
    """ This should be called by all scripts prepared by liftoff.

        python script.py results/something/cfg.yaml

        in your script.py

          if __name__ == "__main__":
              from liftoff import parse_opts()
              main(parse_opts())
    """

    opt_parser = OptionParser("liftoff", ["config_path", "session_id", "results_path"])
    opts = opt_parser.parse_args()

    if opts.results_path != "./results":
        change_out_dir = opts.results_path
    else:
        change_out_dir = None

    config_path = opts.config_path
    with open(opts.config_path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    opts = dict_to_namespace(config_data)

    if not hasattr(opts, "out_dir"):
        opts.out_dir = f"results/experiment_{os.path.dirname(config_path)}"
        opts.run_id = 1
    if check_out_dir and not os.path.isdir(opts.out_dir):  # pylint: disable=no-member
        os.mkdir(opts.out_dir)
        print(f"New out_dir created: {opts.out_dir}")
    elif change_out_dir is not None:
        opts.out_dir = change_out_dir
    else:
        print(f"Existing out_dir: {opts.out_dir}")

    return opts


def set_seed(seed: int, use_cuda: bool, cuda_deterministic: bool = False) -> None:
    """ Set seed for different tools """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if use_cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False


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


def setup_loggers(args: Namespace) -> str:
    """ Load wandb - to args.project_name & with an experiment_name. if logging is activated"""
    import wandb
    from LambdaZero.utils import get_external_dirs

    datasets_dir, programs_dir, summaries_dir = get_external_dirs()

    # Load loggers and setup results path
    use_wandb = args.use_wandb

    if use_wandb:
        experiment_name = f"{args.full_title}_{args.run_id}"
        with open(f"{summaries_dir}/wandb_key") as f:
            api_key = f.readlines()[0].strip()

        os.environ['WANDB_API_KEY'] = api_key

        ret_init = wandb.init(project=args.project_name, name=experiment_name, dir=args.out_dir)
        wandb.config.update(dict(flatten_cfg(args)))
        print(f"[WANDB] INITIALIZED @ {ret_init.dir}")

    out_dir = args.out_dir
    return out_dir


class SummaryStats:
    """ Keep summary statistics of MIN and MAX values - log to wand summary as well"""
    def __init__(self, log_wandb=False):
        self.mem = dict()  # type: Dict[List[int, int]] # dict of [min, max] values
        self.log_wand = log_wandb

    def update(self, new_values: dict) -> Dict:
        mem = self.mem
        update_keys = dict()

        for k, v in new_values.items():
            if k not in mem:
                mem[k] = [v, v]
                update_keys[k] = [True, True]
            else:
                min_m, max_m = mem[k]
                if v <= min_m:
                    update_keys[k] = [True, False]
                    mem[k][0] = v
                elif v >= max_m:
                    update_keys[k] = [False, True]
                    mem[k][1] = v

        if self.log_wand:
            for k, v in update_keys.items():
                if v[0]:
                    wandb.run.summary[f"{k}_MIN"] = mem[k][0]
                if v[1]:
                    wandb.run.summary[f"{k}_MAX"] = mem[k][1]
        return update_keys


def get_stats(values, name, empty_val: float = 0) -> Dict:
    if len(values) == 0:
        values = [empty_val]

    return {
        f"{name}_min": np.min(values),
        f"{name}_max": np.max(values),
        f"{name}_mean": np.mean(values),
        f"{name}_median": np.median(values),
        f"{name}_std": np.std(values),
    }


class AllTimeTop:
    """ Keep buffers for best scores """
    def __init__(self, all_time_topk: List[int] = [10, 100, 1000], order_ascending=True,
                 name: str = "score"):
        self._all_time_topk = all_time_topk
        # Compare new score to worst in buffer
        self._insert = getattr(bisect, "insort_right" if order_ascending else "insort_left")
        self._pop = 0 if order_ascending else -1
        self._best_scores_buffer = {x: [] for x in all_time_topk}
        self._name = name

    def update(self, values: List[float], **kwargs):
        for new_value in values:
            for topk in self._all_time_topk:
                tokv = self._best_scores_buffer[topk]
                self._insert(tokv, new_value)
                if len(tokv) > topk:
                    tokv.pop(self._pop)

    def log(self) -> dict:
        info = {}
        for topk, topkv in self._best_scores_buffer.items():
            info[f"all_time_top{topk}_{self._name}_count"] = len(topkv)
            if len(topkv) > 0:
                info.update(get_stats(topkv, f"all_time_top{topk}_{self._name}"))

        return info


class NumModes:
    def __init__(self, tanimoto_thr=0.7):
        self.tanimoto_thr = tanimoto_thr

        self.modes = []

    def __call__(self, batch):
        # add one mode if needed
        start = 0
        if len(self.modes) == 0:
            self.modes.append(Chem.RDKFingerprint(batch[0]))
            start = 1

        for mol in batch[start:]:
            fp = Chem.RDKFingerprint(mol)
            sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))

            if all(sims < self.tanimoto_thr):
                self.modes.append(fp)

        return len(self.modes)


class TopKNodes:
    def __init__(self, tanimoto_thr=0.7, topk=100, name="score"):
        self.tanimoto_thr = tanimoto_thr
        self.topk = topk
        self._name = name

    def __call__(self, batch, scores, ordered):
        if not ordered:
            raise NotImplementedError

        # add one mode if needed
        modes = []
        modes_score = []
        topk = self.topk
        tanimoto_thr = self.tanimoto_thr

        modes.append(Chem.RDKFingerprint(batch[0]))
        modes_score.append(scores[0])

        idx = 1

        while len(modes) < topk and idx < len(batch):
            mol = batch[idx]
            fp = Chem.RDKFingerprint(mol)
            sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, modes))

            if all(sims < tanimoto_thr):
                modes.append(fp)
                modes_score.append(scores[idx])

            idx += 1

        if len(modes_score) >= topk:
            return get_stats(modes_score, f"all_time_top{topk}nodes_{self._name}")
        else:
            print(f"Not enough nodes {len(modes_score)}")

        return {}


class AllTimeTopWithModes:
    """ Keep buffers for best scores """
    def __init__(self, all_time_topk: List[int] = [1000], order_ascending=True,
                 name: str = "score", tanimoto_thr: float = 0.7):
        self._all_time_topk = all_time_topk
        # Compare new score to worst in buffer
        self._order_ascending = order_ascending
        self._pop = 0 if order_ascending else -1
        self._tanimoto_thr = tanimoto_thr
        self._min_calc = np.argmin if order_ascending else np.argmax
        self._compare = operator.lt if order_ascending else operator.gt
        self._best_scores_buffer = {x: np.zeros(x) for x in all_time_topk}
        self._best_scores_argmin = {}
        self._best_mol_buffer = {x: [] for x in all_time_topk}
        self._name = name
        self._added_cnt = {x: 0 for x in all_time_topk}
        self._over_th = {x: dict() for x in all_time_topk}

    def update(self, values: List[float], mols: List[Any]):
        opc = self._compare
        mcalc = self._min_calc

        for new_value, new_mol in zip(values, mols):
            for topk in self._all_time_topk:
                tokv = self._best_scores_buffer[topk]
                tokvargmin = self._best_scores_argmin
                tokm = self._best_mol_buffer[topk]

                if self._added_cnt[topk] >= topk:
                    min_idx = tokvargmin.get(topk, mcalc(tokv))  # First time requesting argmin
                    if opc(tokv[min_idx], new_value):
                        # we need to add to buffer
                        tokv[min_idx] = new_value
                        tokm[min_idx] = new_mol

                        tokvargmin[topk] = mcalc(tokv)  # Update argmin only when pop out list
                else:
                    tokv[self._added_cnt[topk]] = new_value
                    self._added_cnt[topk] += 1
                    tokm.append(new_mol)

    def log(self) -> dict:
        info = {}
        for topk, topkv in self._best_scores_buffer.items():
            topkv = topkv[:self._added_cnt[topk]]
            info[f"all_time_top{topk}_{self._name}_count"] = len(topkv)
            if len(topkv) > 0:
                info.update(get_stats(topkv, f"all_time_top{topk}_{self._name}"))

                modes_class = NumModes(tanimoto_thr=self._tanimoto_thr)
                order = np.argsort(self._best_scores_buffer[topk])
                # Order from best to worst so we can select mode for the best mol
                order = order[::-1] if self._order_ascending else order
                buff = self._best_mol_buffer[topk]
                n_modes = modes_class([buff[ix] for ix in order])

                info[f"all_time_top{topk}_{self._name}_modesf"] = float(n_modes) / len(topkv)

        return info


class MultiAllTimeTop:
    def __init__(self, all_time_topk, mode_count, order_ascending, tanimoto_thr):
        self._tops = []
        for topk, do_modes in zip(all_time_topk, mode_count):
            if do_modes:
                self._tops.append(
                    AllTimeTopWithModes(
                        all_time_topk=[topk], order_ascending=order_ascending,
                        tanimoto_thr=tanimoto_thr
                    )
                )
            else:
                self._tops.append(AllTimeTop(all_time_topk=[topk], order_ascending=order_ascending))

    def update(self, values: List[float], mols: List[Any]):
        for xtop in self._tops:
            xtop.update(values, mols=mols)

    def log(self) -> dict:
        info_dict = dict()
        for xtop in self._tops:
            info_dict.update(xtop.log())

        return info_dict


class LogTopStats:
    def __init__(self,
                 topk: int = 100,
                 all_time_topk: List[int] = [10, 100, 1000],
                 all_time_topk_mode_count: int = [100],
                 tanimoto_thr: float = 0.7,
                 score_keys: Tuple[str, ...] = ("proxy", "qed", "synth", "dockscore", "dscore"),
                 order_key: str = "score",
                 order_ascending: bool = True,
                 unique_key: str = "res_molecule",
                 transform_info=None,
                 filter_candidates: dict = {"qed": 0.3, "synth": 4.},
                 save_score_buffer_size=10000,
                 out_dir: str = None):

        assert len(set(all_time_topk_mode_count) - set(all_time_topk)) == 0, "Must have the topk with mode in topk list"
        log_topk_with_modes = [True if itop in all_time_topk_mode_count else False for itop in all_time_topk]

        self._topk = topk
        self._all_time_topk_mode_count = all_time_topk_mode_count
        self._all_time_topk = all_time_topk
        self._tanimoto_thr = tanimoto_thr
        self._score_keys = score_keys
        self.do_dockscore = do_dockscore = "dockscore" in score_keys
        self._out_dir = f"{out_dir}/log_top_stats.npy" if out_dir is not None else None

        self._order_key = order_key
        self._seen_mol = set()
        self._saved_mol = []
        self._save_score_buffer_size = save_score_buffer_size
        self._save_keys = list(score_keys) + [order_key, unique_key, "mol"]
        self._collected_mol = 0
        self._new_received = 0
        self._new_have_score = 0
        self._new_good = 0
        self._new_empty = 0
        self._new_smi = 0

        self._new_info = []
        self._order_ascending = order_ascending
        self._compare = operator.lt if order_ascending else operator.gt
        self._unique_key = unique_key
        self._filter_candidates = filter_candidates
        if transform_info is None and do_dockscore:
            transform_info = TransformCompose([
                TransformInfoOracle(), TransformInfoDiscounted()
            ])
        self._transform_info = transform_info

        # self._all_time_topk_score = AllTimeTop(
        #     all_time_topk=all_time_topk,
        #     order_ascending=order_ascending
        # )
        self._all_time_topk_score = MultiAllTimeTop(
            all_time_topk=all_time_topk, mode_count=log_topk_with_modes,
            order_ascending=order_ascending, tanimoto_thr=tanimoto_thr
        )

    def reset(self):
        self._new_info.clear()

    def collect(self, infos: List[dict]):
        self._collected_mol += len(infos)
        self._new_received += len(infos)
        for info in infos:
            _id = info.get(self._unique_key, None)
            _sid = info.get(self._order_key, None)

            self._new_empty += len(info) == 0
            if _id is None or _sid is None:
                continue

            self._new_have_score += 1
            if _id not in self._seen_mol:
                self._new_smi += 1

                good = True
                for k, v in self._filter_candidates.items():
                    if info[k] < v:
                        good = False
                        break

                # If new key and respects candidate conditions
                if good:
                    # The buffer will be reduced each log call
                    self._saved_mol.append({kid: info[kid] for kid in self._save_keys})

                    # TODO Should consider seen only the mol we actually calculate GT for (oracle d)
                    self._seen_mol.update([_id])  # TODO Should fix this

                    self._new_info.append(info)
                    self._all_time_topk_score.update([info[self._order_key]], [info["mol"].mol])

        # Cut only if too big
        if len(self._saved_mol) > 2.0 * self._save_score_buffer_size:
            self.reduce_saved_mol()

    def reduce_saved_mol(self, return_ordered: bool = False):
        order_key = self._order_key

        if len(self._saved_mol) > 1.25 * self._save_score_buffer_size or return_ordered:
            scores = [info[order_key] for info in self._saved_mol]
            order = np.argsort(scores)
            # Order from best to worst so we can select mode for the best mol
            order = order[::-1] if self._order_ascending else order
            topsave = order[:self._save_score_buffer_size]
            smol = self._saved_mol

            self._saved_mol = [smol[ix] for ix in topsave]

        return self._saved_mol

    def log(self):
        if self._new_received == 0:
            self._new_received = 0.000001

        logs = dict({
            f"top{self._topk}_received": self._new_received,
            f"top{self._topk}_good_count": len(self._new_info),
            f"top{self._topk}_good_f": len(self._new_info) / self._new_received,
            f"top{self._topk}_with_score_f": self._new_have_score / self._new_received,
            f"top{self._topk}_new_smi_f": self._new_smi / self._new_received,
            f"top{self._topk}_empty_f": self._new_empty / self._new_received,
            f"top{self._topk}_seen_mol": len(self._seen_mol),
            f"top{self._topk}_collected_mol": self._collected_mol,
        })
        logs.update(self._all_time_topk_score.log())

        if len(self._new_info) > self._topk:
            order_scores = [x[self._order_key] for x in self._new_info]
            sort_idx = np.argsort(order_scores)
            topk_idx = sort_idx[-self._topk:] if self._order_ascending else sort_idx[:self._topk]

            log_info = [self._new_info[x] for x in topk_idx]  # type: List[dict]

            # TODO Should consider seen only the mol we actually calculate GT for
            # # Add logged molecules to seen set
            # for info in log_info:
            #     _id = info.get(self._unique_key, None)
            #     if _id is not None and _id not in self._seen_mol:
            #         self._seen_mol.update([_id])

            if self._transform_info is not None:
                log_info = self._transform_info(log_info)

            for k in self._score_keys:
                scores = [x[k] for x in log_info if k in x and x[k] is not None]

                if len(scores) > 0:
                    logs.update(get_stats(scores, f"top{self._topk}_{k}"))
                else:
                    print(f"No top{self._topk} scores for {k}")

                # Should log this, maybe for e.g. dockscore is None for some of the molecules
                logs.update(dict({f"top{self._topk}_{k}_count": len(scores)}))

            self.reset()

        # Cut saved mol & order & get topKNodes
        ordered_info = self.reduce_saved_mol(return_ordered=True)
        ordered_mol, ordered_scores = [], []
        for info in ordered_info[:self._save_score_buffer_size]:
            ordered_mol.append(info["mol"].mol)
            ordered_scores.append(info[self._order_key])

        for topk_nodes in self._all_time_topk_mode_count:
            calc_nodes = TopKNodes(tanimoto_thr=self._tanimoto_thr, topk=topk_nodes)
            topkmodes_scores = calc_nodes(ordered_mol, ordered_scores, ordered=True)
            # topkmodes_score can be None if not enough top found
            logs.update(topkmodes_scores)

        print(logs)

        self._new_received = 0
        self._new_have_score = 0
        self._new_empty = 0
        self._new_smi = 0

        # Save stuff
        if self._out_dir is not None:
            np.save(self._out_dir, {
                "unique_mols": self._saved_mol,
                # "save_th": self._save_all_score_th,
                "unique_key": self._unique_key,
            })
        return logs


class LogStatsTrain:
    def __init__(self,
                 r_buffer_size: int = 1000,
                 score_keys: Tuple[str] = ("proxy", "qed", "synth"),
                 unique_key: str = "res_molecule",
                 score_key: str = "score",
                 filter_candidates: dict = {"qed": 0.3, "synth": 4.}):
        self._seen_mol = set()
        self._episode_rewards = deque(maxlen=r_buffer_size)
        self._update_act_fails = 0
        self._update_ep_cnt = 0
        self._update_step_cnt = 0
        self._ep_no_scores = 0
        self._update_el_len = []
        self._new_mol_r = []
        self._new_mol_info = []
        self._new_mol_scores = dict({k: [] for k in score_keys})
        self._score_keys = score_keys
        self._unique_key = unique_key
        self._score_key = score_key
        self._filter_candidates = filter_candidates
        for k in filter_candidates.keys():
            assert k in score_keys, f"Must have score key for filer key: {k}"

    def reset(self):
        self._update_act_fails = 0
        self._ep_no_scores = 0
        self._update_ep_cnt = 0
        self._update_step_cnt = 0
        self._new_mol_r = list()
        self._new_mol_info = list()
        self._update_el_len = list()
        self._episode_rewards = list()
        for k, v in self._new_mol_scores.items():
            v.clear()

    def collect_new_batch(self, obs, reward, done, infos):
        self._update_step_cnt += len(obs)
        new_mol_info = []
        for iobs, info in enumerate(infos):
            if done[iobs]:
                self._update_el_len.append(info["num_steps"])
                self._update_ep_cnt += 1
                self._episode_rewards.append(reward[iobs])
                next_mol = info.get(self._unique_key, None)
                if next_mol is None:
                    self._update_act_fails += 1
                elif self._score_key not in info:
                    self._ep_no_scores += 1
                else:
                    if next_mol not in self._seen_mol:
                        self._new_mol_info.append(infos[iobs])
                        new_mol_info.append(infos[iobs])
                        for k, v in self._new_mol_scores.items():
                            v.append(infos[iobs].get(k))
                        self._seen_mol.update([next_mol])
        return new_mol_info

    def log(self) -> Tuple[dict, List[dict]]:
        logs = get_stats(self._episode_rewards, "ep_r", -100)
        logs.update(get_stats(self._update_el_len, "ep_len"))

        logs.update({
            "ep_no_score_count": self._ep_no_scores,
            "ep_no_score_f": self._ep_no_scores / self._update_ep_cnt,
            "act_fails": self._update_act_fails,
        })

        new_mol_scores = self._new_mol_scores
        all_mol = list(range(len(self._new_mol_info)))
        # calculate candidates indxs

        cand_mol = []
        if len(all_mol) > 0:
            cand_mol = set(all_mol)
            for k, v in self._filter_candidates.items():
                idxs = [i for i in all_mol if new_mol_scores[k][i] is not None and new_mol_scores[k][i] >= v]
                cand_mol = cand_mol.intersection(idxs)
            cand_mol = list(cand_mol)

        for score_idxs, set_name in [[all_mol, "new_all"], [cand_mol, "new_mol"]]:
            logs.update({
                f"ep_{set_name}_f": len(score_idxs) / self._update_ep_cnt,
                f"ep_{set_name}_count": len(score_idxs),
            })

            # Get requested scores
            for k, v in new_mol_scores.items():
                scores = [x for ix, x in enumerate(v) if x is not None and ix in score_idxs]
                if len(scores) > 0:
                    logs.update(get_stats(scores, f"{set_name}_{k}"))
                else:
                    print(f"No scores for {set_name}_{k}")

        new_mol_info = self._new_mol_info
        self.reset()
        return logs, new_mol_info


class FakeRemoteLog:
    def __init__(self):
        self._last_log = None
        self.log = Namespace(remote=self.remote)

    def remote(self, log):
        self._last_log = log


if __name__ == "__main__":
    import random
    import string
    import pprint

    letters = string.ascii_lowercase
    log_stats = LogTopStats(topk=100, unique_key="smiles", score_keys=("proxy", "qed", "synth"))

    for i in range(10000):
        proxy = np.random.rand() * -1
        info = {
            "smiles": ''.join(random.choice(letters) for i in range(10)),
            "proxy": proxy,
            "qed": np.random.rand() * 10,
            "synth": np.random.rand() * 10,
            "score": proxy * -1,
        }
        log_stats.collect([info])

    # printing lowercase

    pprint.pprint(log_stats.log())
