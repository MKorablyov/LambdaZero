from argparse import Namespace
from typing import Callable
import torch

from .base_proxy import BaseProxy

from . import proxy_example

PROXYS = {
    "ProxyExample": proxy_example.ProxyExample
}


def get_proxy(cfg: Namespace, **kwargs) -> BaseProxy:
    assert hasattr(cfg, "name") and cfg.name in PROXYS, "Please provide a valid PROXY name."
    return PROXYS[cfg.name](cfg, **kwargs)


def load_proxy(args: Namespace):
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path)

    load_args = checkpoint["config"]

    assert args.name == load_args.name, \
        f"Must have same name {args.name} != {checkpoint['name']}"

    proxy = get_proxy(load_args)
    proxy.load(load_args, checkpoint)
    return proxy
