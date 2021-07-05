from argparse import Namespace
import yaml
import os
from typing import List, Dict, Tuple, Any
import torch
import numpy as np
import random
import wandb
from collections import deque
from liftoff import OptionParser, dict_to_namespace
from lightrl.env.transforms import TransformInfoDiscounted, TransformCompose, TransformInfoOracle


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

        wandb.init(project=args.project_name, name=experiment_name)
        wandb.config.update(dict(flatten_cfg(args)))

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


class LogTopStats:
    def __init__(self,
                 topk: int = 100,
                 score_keys: Tuple[str] = ("proxy", "qed", "synth", "dockscore", "dscore"),
                 order_key: str = "score",
                 order_ascending: bool = True,
                 unique_key: str = "res_molecule",
                 transform_info=None,
                 filter_candidates: dict = {"qed": 0.3, "synth": 4.}):

        self._topk = topk
        self._score_keys = score_keys
        self._order_key = order_key
        self._seen_mol = set()
        self._new_info = []
        self._order_ascending = order_ascending
        self._unique_key = unique_key
        self._filter_candidates = filter_candidates
        if transform_info is None:
            transform_info = TransformCompose([
                TransformInfoOracle(), TransformInfoDiscounted()
            ])
        self._transform_info = transform_info

    def reset(self):
        self._new_info.clear()

    def collect(self, infos: List[dict]):
        for info in infos:
            _id = info.get(self._unique_key, None)
            if _id is not None and _id not in self._seen_mol:
                good = True
                for k, v in self._filter_candidates.items():
                    if info[k] < v:
                        good = False
                        break
                if good:
                    self._new_info.append(info)

    def log(self):
        logs = dict({f"top{self._topk}_count": len(self._new_info)})

        if len(self._new_info) > self._topk:
            order_scores = [x[self._order_key] for x in self._new_info]
            sort_idx = np.argsort(order_scores)
            topk_idx = sort_idx[-self._topk:] if self._order_ascending else sort_idx[:self._topk]

            log_info = [self._new_info[x] for x in topk_idx]  # type: List[dict]

            # Add logged molecules to seen set
            for info in log_info:
                _id = info.get(self._unique_key, None)
                if _id is not None and _id not in self._seen_mol:
                    self._seen_mol.update([_id])

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
