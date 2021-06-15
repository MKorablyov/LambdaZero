import os
import configparser
import numpy as np
import torch
from torch_geometric.utils import remove_self_loops
from rdkit.Chem import AllChem
import pickle


def get_external_dirs():
    """Locate in the filesystem external programs/folders essential for LambdaZero execution
    :return:
    datasets_dir (datasets),
    programs_dir (external program binaries),
    summaries_dir (where to put outputs/model summaries)
    """
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = configparser.ConfigParser()

    if not os.path.exists(os.path.join(ROOT, 'external_dirs.cfg')):
        raise ImportError("Locator file is missing external_dirs.cfg is missing." 
                          "This file, individual for each user should be a part of installation. Please, follow installation instructions in README.md")

    config.read(os.path.join(ROOT, 'external_dirs.cfg'))

    if config["dir"]["datasets"].startswith("/"):
        datasets_dir = config["dir"]["datasets"]
    else:
        datasets_dir = os.path.join(ROOT, config["dir"]["datasets"])
    if config["dir"]["programs"].startswith("/"):
        programs_dir = config["dir"]["programs"]
    else:
        programs_dir = os.path.join(ROOT, config["dir"]["programs"])
    if config["dir"]["summaries"].startswith("/"):
        summaries_dir = config["dir"]["summaries"]
    else:
        summaries_dir = os.path.join(ROOT, config["dir"]["summaries"])

    return datasets_dir, programs_dir, summaries_dir


def dock_metrics(info):
    """ Report custom metrics for each episode in RayRllib
    :param info: episode info
    :return:
    """
    env_info = list(info["episode"]._agent_to_last_info.values())[0]
    episode = info["episode"]
    episode.user_data["molecule"] = env_info["molecule"]
    # print(episode.hist_data)
    for key, value in env_info["log_vals"].items():
        episode.custom_metrics[key] = value


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


def uniform_sample(data, nsamples, nbins=20, nmargin=1, bin_low=None, bin_high=None):
    data = np.asarray(data, dtype=np.float32)
    assert len(data.shape) == 1, "requires flat array"
    ndata = data.shape[0]
    if bin_low is None:
        bin_low = data[np.argpartition(data, nmargin)[nmargin]].max()
    if bin_high is None:
        bin_high = data[np.argpartition(-data, nmargin)[nmargin]].min()
    print("partitioning data with bin low high", bin_low, bin_high)
    # print("smax smin:", smax,data.max(), smin, data.min())
    bins = np.linspace(bin_low, bin_high, num=nbins+1)[:-1]
    bin_data = np.digitize(data, bins) - 1
    # todo: I don't think my sampler actually works properly for this distributuion of 100M
    n_per_bin = nsamples // nbins
    sele_idxs = [[] for _ in range(nbins)]
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


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):

        if len(x.shape) > 0:
            batch_count = x.shape[0]
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
        else:
            batch_mean = x
            batch_var = 0
            batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


datasets_dir, programs_dir, summaries_dir = get_external_dirs()
pca_path = os.path.join(datasets_dir, "brutal_dock/mpro_6lze/raw/pca.pkl")
pca_cache = [None]


def molecule_pca(mol):
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2)]
    mat = []
    for fp in fps:
        bits = fp.ToBitString()
        bitsvec = [int(bit) for bit in bits]
        mat.append(bitsvec)
    mat = np.array(mat)

    pca = pickle.load(open(pca_path, 'rb'))
    scaled_data = pca.transform(mat)
    log_vals = {"PC1": scaled_data[0][0], "PC2": scaled_data[0][1]}
    return log_vals


def logP(mu, sigma, x):
    """
    Estimate log likelihood of an estimator
    :param mu: estimated mu
    :param sigma: estimated sigma
    :param x: ground truth
    :return:
    """
    return -np.log(sigma * (2 * np.pi)**0.5) - 0.5 * (((x - mu) / sigma) ** 2)
