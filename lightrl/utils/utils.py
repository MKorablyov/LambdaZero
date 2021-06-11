from argparse import Namespace
import yaml
import os
from typing import List
import torch
import numpy as np
import random
import wandb

from liftoff import OptionParser, dict_to_namespace


def flatten_cfg(cfg: Namespace):
    lst = []
    for key, value in cfg.__dict__.items():
        if isinstance(value, Namespace):
            for key2, value2 in flatten_cfg(value):
                lst.append((f"{key}.{key2}", value2))
        else:
            lst.append((key, value))
    return lst


def update_cfg(base_cfg: Namespace, new_cfg: Namespace):
    """ Inplace update cfg """
    for k, v in new_cfg.__dict__.items():
        if isinstance(base_cfg.__dict__.get(k, None), Namespace):
            update_cfg(base_cfg.__dict__[k], v)
        else:
            base_cfg.__dict__[k] = v


def add_to_cfg(cfg: Namespace, subgroups: List[str], new_arg: str, new_arg_value) -> None:
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


def set_seed(seed: int, use_cuda: bool, cuda_deterministic: bool = False):
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


def setup_loggers(args: Namespace):
    import wandb
    import csv
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


class SummaryStats:
    def __init__(self, log_wand=False):
        self.mem = dict()  # dict of [min, max] values
        self.log_wand = log_wand

    def update(self, new_values: dict):
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
