import time
import os.path as osp
import configparser
import numpy as np
import torch
from torch_geometric.utils import remove_self_loops

from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import pickle as pk


def get_external_dirs():
    """Locate in the filesystem external programs/folders essensial for LambdaZero execution
    :return:
    datasets_dir (datasets),
    programs_dir (external program binaries),
    summaries_dir (where to put outpus/model summaries)
    """
    ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    config = configparser.ConfigParser()

    if not osp.exists(osp.join(ROOT, 'external_dirs.cfg')):
        raise ImportError("Locator file is missing. Can't locate third party dependencies. "
                          "Please, run install_prog_data.sh which will install datasets and third party dependencies "
                          "and also create external_dirs.cfg  in process needed to locate to these softwares."
                          "In case you already have datasets and programs installed, you could create "
                          "external_dirs.cfg manually to point to installations.")

    config.read(osp.join(ROOT, 'external_dirs.cfg'))

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



# log_env_info
def dock_metrics(info):
    """ Report custom metrics for each episode in RayRllib
    :param info: episode info
    :return:
    """
    env_info = list(info["episode"]._agent_to_last_info.values())[0]
    episode = info["episode"]
    episode.user_data["molecule"] = env_info["molecule"]
    #print(episode.hist_data)
    for key, value in env_info["log_vals"].items():
        episode.custom_metrics[key] = value


class MeanVarianceNormalizer:
    def __init__(self, mean_and_variance):
        self.mean = mean_and_variance[0]
        self.variance = mean_and_variance[1]

    def normalize(self, x):
        x_norm = (x - self.mean) / self.variance
        return x_norm

    def unnormalize(self, x_norm):
        x = (x_norm * self.variance) + self.mean
        return x


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device
        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

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


datasets_dir, programs_dir, summaries_dir = get_external_dirs()
pca_path = osp.join(datasets_dir, "brutal_dock/mpro_6lze/raw/pca.pkl")

def molecule_pca(mol):
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2)]
    mat = []
    for fp in fps:
        bits = fp.ToBitString()
        bitsvec = [int(bit) for bit in bits]
        mat.append(bitsvec)
    mat = np.array(mat)

    pca = pk.load(open(pca_path, 'rb'))
    scaled_data = pca.transform(mat)
    log_vals = {"PC1": scaled_data[0][0], "PC2": scaled_data[0][1]}
    return (log_vals)