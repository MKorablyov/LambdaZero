import time
import os.path as osp
import configparser
import torch as th
from torch_geometric.utils import remove_self_loops


def get_external_dirs():
    """Locate in the filesystem external programs/folders essensial for LambdaZero execution
    :return:
    datasets_dir (datasets),
    programs_dir (external program binaries),
    summaries_dir (where to put outpus/model summaries)
    """
    ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    config = configparser.ConfigParser()
    config.read(osp.join(ROOT, 'setup.cfg'))

    if config["dir"]["datasets"].startswith("/"):
        datasets_dir = config["dir"]["datasets"]
    else:
        datasets_dir = osp.join(ROOT, config["dir"]["datasets"])
    if config["dir"]["programs"].startswith("/"):
        programs_dir = config["dir"]["programs"]
    else:
        programs_dir = osp.join(ROOT, config["dir"]["programs"])
    if config["dir"]["summaries"].startswith("/"):
        summaries_dir = config["dir"]["summaries"]
    else:
        summaries_dir = osp.join(ROOT, config["dir"]["summaries"])

    return datasets_dir, programs_dir, summaries_dir


def dock_metrics(info):
    """
    Report custom metrics for each episode in RayRllib
    :param info: episode info
    :return:
    """
    env_info = list(info["episode"]._agent_to_last_info.values())[0]
    episode = info["episode"]
    episode.custom_metrics["reward"] = env_info["reward"]
    episode.custom_metrics["QED"] = env_info["QED"]
    episode.custom_metrics["discounted_reward"] = env_info["discounted_reward"]



# class Normalize(object):
#     def __init__(self, target, target_norm):
#         self.target = target
#         self.target_norm = target_norm
#
#     def __call__(self, data):
#         # Specify target.
#         y = getattr(data, self.target)
#         y = (y - self.target_norm[0]) / self.target_norm[1]
#         data.y = y
#         return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device
        row = th.arange(data.num_nodes, dtype=th.long, device=device)
        col = th.arange(data.num_nodes, dtype=th.long, device=device)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = th.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        return data
