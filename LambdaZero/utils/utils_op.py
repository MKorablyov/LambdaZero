import time
import os.path as osp
import configparser
import numpy as np
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

def uniform_sample(data,nsamples,nbins=20,nmargin=1,bin_low=None,bin_high=None):
    data = np.asarray(data,dtype=np.float)
    assert len(data.shape)==1,"requires flat array"
    ndata = data.shape[0]
    if bin_low == None:
        bin_low = data[np.argpartition(data, nmargin)[nmargin]].max()
    if bin_high == None:
        bin_high = data[np.argpartition(-data, nmargin)[nmargin]].min()
    print("partitioning data with bin low high", bin_low, bin_high)
    #print("smax smin:", smax,data.max(), smin, data.min())
    bins = np.linspace(bin_low,bin_high,num=nbins+1)[:-1]
    bin_data = np.digitize(data,bins) - 1
    # todo: I don't think my sampler actually works properly for this distributuion of 100M
    n_per_bin = nsamples // nbins
    sele_idxs = [[] for i in range(nbins)]
    data_idxs = np.arange(ndata)
    np.random.shuffle(data_idxs)
    for data_idx in data_idxs:
        if len(sele_idxs[bin_data[data_idx]]) < n_per_bin:
            sele_idxs[bin_data[data_idx]].append(data_idx)
    print("sele bincounts", [len(sele_idxs[i]) for i in range(nbins)])
    # print("sele binrange", [(i, ":", max(sele_idxs[i]), min(sele_idxs[i])) for i in range(nbins)])
    sele_idxs = np.concatenate([np.asarray(sele_idxs[i], dtype=np.int) for i in range(nbins)])
    # print("min max bin data", bin_data.min(),bin_data.max())
    # bin_counts = Counter(bin_data)
    # bin_counts = np.asarray(list(bin_counts.values()),dtype=np.int32)[np.argsort(list(bin_counts.keys()))]
    # print("bin counts",  bin_counts)
    # bin_probs = 1./bin_counts
    # data_probs = bin_probs[bin_data]
    # data_probs = data_probs / np.sum(data_probs)
    # print(np.sum(data_probs))
    # idx = np.random.choice(ndata,nsamples,replace=True,p=data_probs)
    return sele_idxs
