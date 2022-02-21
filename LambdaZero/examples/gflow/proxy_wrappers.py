""" Classes to calculate other scores"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
import torch
from typing import List, Any
import os.path as osp
from collections import Counter, deque
import glob

from LambdaZero.models import ChempropWrapper_v1
from chemprop.features import BatchMolGraph, MolGraph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from argparse import Namespace
from LambdaZero.contrib.functional import elu2
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments.molMDP import BlockMoleculeData
from LambdaZero.examples.proxys import load_proxy
from scipy import stats
import math

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def satlins(x, cutoff0, cutoff1, eps_min=0.01):
    "shifted saturated linearity activation function _/-"
    x = (x - cutoff0) / (cutoff1 - cutoff0)
    x = min(max(eps_min, x), 1.0)  # relu to maxout at 1
    return x


def elu2(x, x_shift=-1., y_shift=1., epsilon=math.e):
    "this is elu which is always positive and 1 is 1 with default settings"
    x = np.asarray(x,dtype=np.float32) + x_shift
    act = []
    for val in x:
        if val > 0:
            act.append(float(y_shift + val))
        else:
            act.append(float(y_shift + (epsilon ** val-1)))
    return act


def score_clipped(_synth_cut, _qed_cut, infos, idxs, ):
    clip_qed = np.array([
        satlins(q, _qed_cut[0], _qed_cut[1]) for q in [infos[x]["qed"] for x in idxs]
    ])

    clip_synth = np.array([
        satlins(q, _synth_cut[0], _synth_cut[1]) for q in
        [infos[x]["synth"] for x in idxs]
    ])


class QEDEstimator:
    def __init__(self):
        pass

    def eval(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        qed = QED.qed(mol)
        return qed

    def __call__(self, smiles: List[str]):
        ret = np.array([self.eval(x) for x in smiles])
        return ret


class ParallelSynth(torch.nn.Module):
    def __init__(self, synth_config=None, use_cuda=True):
        super(ParallelSynth, self).__init__()

        if synth_config is None:
            from LambdaZero.environments.block_mol_v3 import synth_config

        self._synth_net = ChempropWrapper_v1(synth_config, use_cuda=use_cuda)

        self._synth_net.model.eval()

    @torch.no_grad()
    def __call__(self, mols):
        mols = BatchMolGraph([MolGraph(x) for x in mols])
        return self.eval_batch(mols)

    @torch.no_grad()
    def eval_batch(self, mols):
        model = self._synth_net.model
        features = self._synth_net.predict_config["features_generator"]

        with torch.no_grad():
            preds = model(mols, features)

        preds = preds.data.cpu().numpy()
        if self._synth_net.scaler is not None:
            preds = self._synth_net.scaler.inverse_transform(preds)

        return preds.reshape(-1)


class CandidateWrapper(torch.nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super(CandidateWrapper, self).__init__()

        device = args.device

        self.proxy = load_proxy(args.proxy)
        self.synth_net = ParallelSynth(use_cuda=not (device.type == "cpu"))
        self.qed = QEDEstimator()
        self._default_score = args.default_score
        self.qed_th = getattr(args, "qed_th", 0.3)
        self.synth_th = getattr(args, "synth_th", 4.)

        self._seen = dict()
        self._seen_nan_count = 0
        self._seen_p = []
        self._seen_scores = deque(maxlen=10000)

    @torch.no_grad()
    def __call__(self, mols: List[BlockMoleculeData]):
        mols = mols if isinstance(mols, list) else [mols]

        res_scores = [self._default_score] * len(mols)
        infos = [{x: None for x in ["proxy", "qed", "synth", "score", "smiles"]} for _ in range(len(mols))]

        mol_mol = [mol.mol for mol in mols]
        good = [mmm is not None for mmm in mol_mol]
        update_mem = []

        for ix, ggg in enumerate(good):
            if ggg:
                infos[ix]["smiles"] = smi = mols[ix].smiles
                if smi in self._seen:
                    infos[ix].update(self._seen[smi][1])
                    res_scores[ix] = self._seen[smi][0]
                    good[ix] = False
                    continue
                else:
                    update_mem.append(ix)

                infos[ix]["qed"] = qed_score = QED.qed(mol_mol[ix])
                if qed_score <= self.qed_th:
                    good[ix] = False
            else:
                self._seen_nan_count += 1

        valid = [x for x, ggg in zip(mol_mol, good) if ggg]
        if len(valid) > 0:
            synth_scores = list(self.synth_net(valid))
            for ix, ggg in enumerate(good):
                if ggg:
                    infos[ix]["synth"] = synth_score = synth_scores.pop(0)
                    if synth_score <= self.synth_th:
                        good[ix] = False

            valid = [x for x, ggg in zip(mols, good) if ggg]
            if len(valid) > 0:
                proxy_scores = list(self.proxy(valid))
                for ix, ggg in enumerate(good):
                    if ggg:
                        infos[ix]["proxy"] = res_scores[ix] = proxy_scores.pop(0)
                        infos[ix]["score"] = res_scores[ix] * -1

                        infos[ix]["mol"] = mols[ix] # .dump() # Should add for proxy calc

        for ix in update_mem:
            self._seen[infos[ix]["smiles"]] = [res_scores[ix], infos[ix]]
            self._seen_p.append(res_scores[ix])

        self._seen_scores += [x * -1 for x in res_scores]

        return res_scores, infos


class ProxyOnlyWrapper(torch.nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super(ProxyOnlyWrapper, self).__init__()

        self.proxy = load_proxy(args.proxy)
        self._default_score = args.default_score
        self._memory = dict()

        self._seen = set()
        self._seen_nan_count = 0
        self._seen_p = []
        self._seen_scores = deque(maxlen=10000)

    @torch.no_grad()
    def __call__(self, mols: List[BlockMoleculeData]):
        mols = mols if isinstance(mols, list) else [mols]

        res_scores = [self._default_score] * len(mols)
        infos = [{x: None for x in ["proxy", "qed", "synth", "score", "smiles"]} for _ in range(len(mols))]

        mol_mol = [mol.mol for mol in mols]
        good = [mmm is not None for mmm in mol_mol]

        valid = [x for x, ggg in zip(mols, good) if ggg]
        if len(valid) > 0:
            proxy_scores = list(self.proxy(valid))

        for ix, ggg in enumerate(good):
            if ggg:
                proxy = proxy_scores.pop(0)

                if mols[ix].smiles not in self._seen:
                    self._seen.update([mols[ix].smiles])
                    self._seen_p.append(proxy)
                self._seen_scores.append(proxy * -1)

                infos[ix]["smiles"] = mols[ix].smiles
                infos[ix]["synth"] = 100  # TODO Hack to not change candidate filering
                infos[ix]["qed"] = 100  # TODO Hack to not change candidate filering
                infos[ix]["proxy"] = res_scores[ix] = proxy
                infos[ix]["score"] = res_scores[ix] * -1
                infos[ix]["mol"] = mols[ix]  # .dump()
            else:
                self._seen_nan_count += 1

        return res_scores, infos


class ProxyQED:
    def __init__(self, args):
        pass

    def __call__(self, mols):
        # QED constraint
        qed = [QED.qed(mol.mol) for mol in mols]
        return qed


class QEDProxyOnlyWrapper(torch.nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super(QEDProxyOnlyWrapper, self).__init__()

        self.proxy = ProxyQED(args)
        self._default_score = args.default_score
        self._memory = dict()

        self._seen = set()
        self._seen_nan_count = 0
        self._seen_p = []
        self._seen_scores = deque(maxlen=10000)

    @torch.no_grad()
    def __call__(self, mols: List[BlockMoleculeData]):
        mols = mols if isinstance(mols, list) else [mols]

        res_scores = [self._default_score] * len(mols)
        infos = [{x: None for x in ["proxy", "qed", "synth", "score", "smiles"]} for _ in range(len(mols))]

        mol_mol = [mol.mol for mol in mols]
        good = [mmm is not None for mmm in mol_mol]

        valid = [x for x, ggg in zip(mols, good) if ggg]
        if len(valid) > 0:
            proxy_scores = list(self.proxy(valid))

        for ix, ggg in enumerate(good):
            if ggg:
                proxy = proxy_scores.pop(0)

                if mols[ix].smiles not in self._seen:
                    self._seen.update([mols[ix].smiles])
                    self._seen_p.append(proxy)
                self._seen_scores.append(proxy * -1)

                infos[ix]["smiles"] = mols[ix].smiles
                infos[ix]["synth"] = 100  # TODO Hack to not change candidate filering
                infos[ix]["qed"] = 100  # TODO Hack to not change candidate filering
                infos[ix]["proxy"] = res_scores[ix] = proxy
                infos[ix]["score"] = res_scores[ix]
                infos[ix]["mol"] = mols[ix]  # .dump()
            else:
                self._seen_nan_count += 1

        return res_scores, infos


class CandidateWrapperSatlin(torch.nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super(CandidateWrapperSatlin, self).__init__()

        self.proxy = load_proxy(args.proxy)
        self._default_score = args.default_score
        self._memory = dict()

        self._seen = set()
        self._seen_nan_count = 0
        self._seen_p = []
        self._seen_scores = deque(maxlen=10000)

        device = args.device
        self.synth_net = ParallelSynth(use_cuda=not (device.type == "cpu"))
        self.qed = QEDEstimator()
        self._default_score = args.default_score
        self.qed_th = getattr(args, "qed_th", 0.3)
        self.synth_th = getattr(args, "synth_th", 4.)
        self.candv = getattr(args, "candv", 0)

        cutoff_min = getattr(args, "cutoff_min", 0.)
        print("cutoff_min", cutoff_min)
        self.qed_cutoff = [cutoff_min, self.qed_th]
        self.synth_cutoff = [cutoff_min, self.synth_th]

    def calc_score_v1(self, info):
        sat_q = satlins(info["qed"], *self.qed_cutoff)
        sat_s = satlins(info["synth"], *self.synth_cutoff)
        res = min(0, info["proxy"]) * sat_q * sat_s
        return res

    @torch.no_grad()
    def __call__(self, mols: List[BlockMoleculeData]):
        mols = mols if isinstance(mols, list) else [mols]

        res_scores = [self._default_score] * len(mols)
        infos = [{x: None for x in ["proxy", "qed", "synth", "score", "smiles"]} for _ in range(len(mols))]

        mol_mol = [mol.mol for mol in mols]
        good = [mmm is not None for mmm in mol_mol]

        valid = [x for x, ggg in zip(mols, good) if ggg]
        if len(valid) > 0:
            proxy_scores = list(self.proxy(valid))

        for ix, ggg in enumerate(good):
            if ggg:
                proxy = proxy_scores.pop(0)

                if mols[ix].smiles not in self._seen:
                    self._seen.update([mols[ix].smiles])
                    self._seen_p.append(proxy)
                self._seen_scores.append(proxy * -1)

                infos[ix]["smiles"] = mols[ix].smiles
                infos[ix]["synth"] = self.synth_net([mol_mol[ix]])[0]
                infos[ix]["qed"] = QED.qed(mol_mol[ix])
                infos[ix]["proxy"] = proxy
                res_scores[ix] = self.calc_score_v1(infos[ix])
                infos[ix]["score"] = res_scores[ix] * -1
                infos[ix]["mol"] = mols[ix]  # .dump()
            else:
                self._seen_nan_count += 1

        return res_scores, infos


class ProxyDebugWrapper(torch.nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super(ProxyDebugWrapper, self).__init__()

        self.proxy = load_proxy(args.proxy)
        self._default_score = args.default_score

        fake_mol = getattr(args, "fake_mol", 100)
        fake_range = getattr(args, "fake_range", [-13, -10])
        self._freq_to_allocate = getattr(args, "allocate_freq", 1.)
        self._fake_scores = np.linspace(*fake_range, fake_mol).tolist()
        self._unallocated_scores = list(self._fake_scores)
        self._seen = set()
        self._seen_scores = Counter()
        self._allocated = {'COC(CO)OCO': -14.0, 'COC(O)c1cccc(C=O)c1': -13.333333333333334, 'COc1cccc(OCCO)c1': -12.666666666666666, 'COCOOCC=O': -12.0}
        self._max_len_blocks = 0


    @torch.no_grad()
    def __call__(self, mols: List[BlockMoleculeData]):
        # print(len(self._unallocated_scores), len(self._allocated), len(self._seen))
        mols = mols if isinstance(mols, list) else [mols]

        res_scores = [self._default_score] * len(mols)
        infos = [{x: None for x in ["proxy", "qed", "synth", "score", "smiles"]} for _ in range(len(mols))]

        mol_mol = [mol.mol for mol in mols]
        good = [mmm is not None for mmm in mol_mol]

        valid = [x for x, ggg in zip(mols, good) if ggg]
        # if len(valid) > 0:
        #     proxy_scores = list(self.proxy(valid))

        for ix, ggg in enumerate(good):
            if ggg:
                smiles = mols[ix].smiles

                if smiles in self._allocated:
                    proxy = self._allocated[smiles]
                else:
                    proxy = self._default_score

                self._seen.update([smiles])
                self._seen_scores.update(np.around([proxy * -1], 1).tolist())
                self._max_len_blocks = max(self._max_len_blocks, len(mols[ix].blockidxs))

                infos[ix]["smiles"] = smiles
                infos[ix]["synth"] = 100  # TODO Hack to not change candidate filering
                infos[ix]["qed"] = 100  # TODO Hack to not change candidate filering

                infos[ix]["proxy"] = res_scores[ix] = proxy
                infos[ix]["score"] = res_scores[ix] * -1
                infos[ix]["mol"] = mols[ix]  # .dump()

        return res_scores, infos


class ProxyDebug131_11(torch.nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super(ProxyDebug131_11, self).__init__()
        source = "/scratch/andrein/Datasets/Datasets/full_mdp_blocks_debug_131_max_steps5.pk_all.pk"


        self.args = args
        self.candidate_score = candidate_score = getattr(args, "candidate_score", False)
        self.proxy = load_proxy(args.proxy)
        self._default_score = args.default_score
        self._memory = dict()

        self._seen = set()
        self._seen_p_uniq = set()
        self._seen_nan_count = 0
        self._seen_p = []
        self._seen_scores = deque(maxlen=10000)
        self._debug_data = data = torch.load(source)

        self._uniq_leaves_infos = uniq_leaves_infos = data["infos"]
        self._uniq_leaves_smi_info = {x["smiles"]: x for x in data["infos"]}

        self.cand_scores = cand_scores = []

        if self.candidate_score:
            device = args.device
            self.synth_net = ParallelSynth(use_cuda=not (device.type == "cpu"))
            self.qed = QEDEstimator()
            self._default_score = args.default_score
            self.qed_th = getattr(args, "qed_th", 0.3)
            self.synth_th = getattr(args, "synth_th", 4.)
            self.candv = getattr(args, "candv", 0)

            print(f"[ProxyDebug131_11] Debug with candidate score v {self.candv}")
            if self.candv == 0:
                self.calc_score = self.calc_score_v0
            elif self.candv == 1:
                self.calc_score = self.calc_score_v1
                self.qed_cutoff = [0.0, self.qed_th]
                self.synth_cutoff = [0, self.synth_th]
            else:
                raise NotImplementedError

            # scores = self.get_scores(self.)
            for ix, info in enumerate(data["infos"]):
                info["proxy"] = self.calc_score(info)
                # debug found based on hard threshold candidate score
                # cand_scores.append(self.calc_score_v0(info))
                cand_scores.append(info["proxy"])

                # import pdb; pdb.set_trace()
        else:
            cand_scores += [info["proxy"] for info in data["infos"]]

        self._seen_p = sorted([x["proxy"] for x in self._uniq_leaves_infos])
        self._leaves_smi = [x["smiles"] for x in self._uniq_leaves_infos]
        self._uniq_leaves_smi = set(self._leaves_smi)
        self._new_found_smi = []

        print(f"[ProxyDebug131_11] Uniq leaves {len(uniq_leaves_infos)} "
              f"with uniq smiles {len(self._uniq_leaves_smi)} "
              f"and uniq scores {len(np.unique(self._seen_p))}")

        unique_smile_score = dict()
        for info in uniq_leaves_infos:
            unique_smile_score[info["smiles"]] = info["proxy"]
        calc_top = unique_smile_score.values() # not seen_p because it contains duplicate smiles
        calc_top = sorted(calc_top)

        self.tops = tops = [np.mean([calc_top[:topk]]) * -1 for topk in [10, 100, 1000]]
        print(f"[ProxyDebug131_11] Best score: {self._seen_p[0] *-1} "
              f"| Top10: {tops[0]} | Top100: {tops[1]} | Top1000: {tops[2]}")

        self.debug_topk = 5000

        top_idxs = np.argsort(self.cand_scores)[:self.debug_topk]
        self.debug_top_scores = set([self._uniq_leaves_infos[ix]["proxy"] for ix in top_idxs])
        self.debug_top_smi = set([self._uniq_leaves_infos[ix]["smiles"] for ix in top_idxs])
        self.debug_top_scores_worst = max(self.debug_top_scores)
        print(f"[ProxyDebug131_11] Top {self.debug_topk} "
              f"uniq smi {len(self.debug_top_smi)} "
              f"& uniq scores {len(self.debug_top_scores)} |"
              f"best score {min(self.debug_top_scores)} | worst score {max(self.debug_top_scores)}")

    def get_dump(self):
        numf = len(glob.glob("/scratch/andrein/Datasets/Datasets/debug_extra/*"))
        dump = f"/scratch/andrein/Datasets/Datasets/debug_extra/new_{numf}.pk"
        return dump

    def found_top(self):
        if len(self._new_found_smi) > 1000:
            torch.save({"new": self._new_found_smi}, self.get_dump())
            self._new_found_smi.clear()

        return len(self._seen.intersection(self.debug_top_smi)) / len(self.debug_top_smi)
        # return len(self._seen_p_uniq.intersection(self.debug_top_scores)) / len(self.debug_top_scores)
        # return len(self._seen_p_uniq) / len(self.debug_top_scores)

    def debug(self, dataset):
        print("RUNNING PROXY DEBUG PREPROCESSING")
        _uniq_smi_r = []
        all_tr_r = []
        all_tr_p = []
        for info in self._uniq_leaves_infos:
            # assert smi == info["smiles"], f"Not same smile {info} vs {smi}"
            true_r = dataset.r2r(dockscore=info["proxy"])
            all_tr_r += [true_r]
            all_tr_p += [info["proxy"]*-1]
            _uniq_smi_r.append(true_r)

        print(f"[ProxyDebug131_11] Direct Histogram R \n {np.histogram(all_tr_r)}")
        print(f"[ProxyDebug131_11] Direct Histogram scores \n {np.histogram(self._seen_p)}")

        self._sum_all_tr_r = sum(all_tr_r)

        num_bins = 32
        quantiles = np.linspace(0, 1, num_bins)[1:-1]
        uniq_scores = [x*-1 for x in self._seen_p]
        if self.candidate_score:
            # TODO Fast and dirty solution in case we use sparse reward and most scores contain 16 (75%)
            uniq_scores = np.unique(uniq_scores)
        score_bin_edges = stats.mstats.mquantiles(uniq_scores, quantiles)
        score_bin_edges = np.array([-np.inf] + score_bin_edges.tolist() + [np.inf])

        # import pdb; pdb.set_trace()
        r_sums, score_edges, binnumber = stats.binned_statistic(all_tr_p, all_tr_r, 'sum', bins=score_bin_edges)
        # import pdb; pdb.set_trace()

        bin_count = np.zeros(num_bins)
        binid, bincnt = np.unique(binnumber, return_counts=True)
        bin_count[binid-1] = bincnt
        bin_count = bin_count.astype(np.int)

        bin_probs = r_sums/self._sum_all_tr_r
        print(f"[ProxyDebug131_11] Scores hist edges\n {np.around(score_edges, 2).tolist()}")
        print(f"[ProxyDebug131_11] Scores hist cnt\n {bin_count.tolist()}")
        print(f"[ProxyDebug131_11] TGT Bin prob (given R transform ) \n {np.around(bin_probs * 100, 2).tolist()}")

        self._tgt_score_edges = score_edges

        # UGLY FOR PLOT
        show_edge = score_edges.copy()
        show_edge[0] = score_edges[1] - 1.
        show_edge[-1] = score_edges[-2] + 1.
        print(f"[ProxyDebug131_11] Show edges", show_edge)

        self._show_tgt_score_edges = show_edge
        self._tgt_r_bin_probs = bin_probs
        self._tgt_probs_zrange = (bin_probs.min(), bin_probs.max()*1.5)

        # GET Eval trajectories
        data = self._debug_data
        if False:
            debug_transitions = []
            for mol_sample, mol_sample_idx in zip(data["mol_samples"], data["mol_samples_idx"]):
                traj = dataset.get_backward_trajectory(mol_sample, _uniq_smi_r[mol_sample_idx])
                debug_transitions += traj
            self._debug_batch = dataset.sample2batch(debug_transitions)

    def get_scores(self, valid_mols: List):
        if self.candidate_score:
            scores = [self._default_score] * len(valid_mols)

            do_synth = []
            do_qed = []
            do_proxy = []
            infos = []
            for ix, mol in enumerate(valid_mols):
                smi = mol.smiles
                if smi in self._uniq_leaves_smi_info:
                    self._uniq_leaves_smi_info[smi] = {x: None for x in ["proxy", "qed", "synth", "score", "smiles"]}
                    self._uniq_leaves_smi_info[smi]["smiles"] = smi

                info = self._uniq_leaves_smi_info[smi]
                if info.get("qed") is None:
                    do_qed.append(ix)
                if info.get("synth") is None:
                    do_synth.append(ix)
                if info.get("proxy") is None:
                    do_proxy.append(ix)
                infos.append(info)

            vals = [QED.qed(valid_mols[ix].mol) for ix in do_qed]
            for ix, vvv in zip(do_qed, vals):
                infos[ix]["qed"] = vvv
            vals = list(self.synth_net([valid_mols[ix].mol for ix in do_synth]))
            for ix, vvv in zip(do_synth, vals):
                infos[ix]["synth"] = vvv

            vals = list(self.proxy([valid_mols[ix] for ix in do_proxy]))
            for ix, vvv in zip(do_proxy, vals):
                infos[ix]["proxy"] = vvv

            # Calculate score
            for ix, info in enumerate(infos):
                scores[ix] = self.calc_score(info)
            return scores
        else:
            return list(self.proxy(valid_mols))

    def calc_score(self, info):
        raise NotImplementedError

    def calc_score_v0(self, info):
        if info["qed"] > self.qed_th and info["synth"] > self.synth_th:
            return info["proxy"]
        return self._default_score

    def calc_score_v1(self, info):
        sat_q = satlins(info["qed"], * self.qed_cutoff)
        sat_s = satlins(info["synth"], * self.synth_cutoff)
        return min(0, info["proxy"]) * sat_q * sat_s


    @torch.no_grad()
    def __call__(self, mols: List[BlockMoleculeData]):
        mols = mols if isinstance(mols, list) else [mols]

        res_scores = [self._default_score] * len(mols)
        infos = [{x: None for x in ["proxy", "qed", "synth", "score", "smiles"]} for _ in range(len(mols))]

        mol_mol = [mol.mol for mol in mols]
        good = [mmm is not None for mmm in mol_mol]

        valid = [x for x, ggg in zip(mols, good) if ggg]
        if len(valid) > 0:
            proxy_scores = self.get_scores(valid) # list(self.proxy(valid))

        for ix, ggg in enumerate(good):
            if ggg:
                proxy = proxy_scores.pop(0)
                # self._seen_p_uniq.update([proxy])
                if proxy <= self.debug_top_scores_worst:
                    self._seen_p_uniq.update([proxy])

                if mols[ix].smiles not in self._seen:
                    self._seen.update([mols[ix].smiles])
                    if mols[ix].smiles not in self._uniq_leaves_smi:
                        self._new_found_smi.append(mols[ix])
                        self._uniq_leaves_smi.update([mols[ix].smiles])

                self._seen_scores.append(proxy * -1)

                infos[ix]["smiles"] = mols[ix].smiles
                infos[ix]["synth"] = 100  # TODO Hack to not change candidate filering
                infos[ix]["qed"] = 100  # TODO Hack to not change candidate filering
                infos[ix]["proxy"] = res_scores[ix] = proxy
                infos[ix]["score"] = res_scores[ix] * -1
                infos[ix]["mol"] = mols[ix]  # .dump()
            else:
                self._seen_nan_count += 1

        return res_scores, infos


PROXY_WRAPPERS = {
    "CandidateWrapper": CandidateWrapper,
    "ProxyOnlyWrapper": ProxyOnlyWrapper,
    "QEDProxyOnlyWrapper": QEDProxyOnlyWrapper,
    "ProxyDebugWrapper": ProxyDebugWrapper,
    "ProxyDebug131_11": ProxyDebug131_11,
    "CandidateWrapperSatlin": CandidateWrapperSatlin,
}
