""" Classes to calculate other scores"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
import torch
from typing import List, Any
import os.path as osp

from LambdaZero.models import ChempropWrapper_v1
from chemprop.features import BatchMolGraph, MolGraph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from argparse import Namespace
from LambdaZero.contrib.functional import elu2
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments.molMDP import BlockMoleculeData
from LambdaZero.examples.proxys import load_proxy

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


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

    def __call__(self, mols):
        mols = BatchMolGraph([MolGraph(x) for x in mols])
        return self.eval_batch(mols)

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
    def __init__(self, args: Namespace):
        super(CandidateWrapper, self).__init__()

        device = args.device

        self.proxy = load_proxy(args.proxy)
        self.synth_net = ParallelSynth(use_cuda=not (device.type == "cpu"))
        self.qed = QEDEstimator()
        self._default_score = args.default_score
        self.qed_th = getattr(args, "qed_th", 0.3)
        self.synth_th = getattr(args, "synth_th", 4.)

    def __call__(self, mols: List[BlockMoleculeData]):
        mols = mols if isinstance(mols, list) else [mols]

        res_scores = [self._default_score] * len(mols)
        infos = [{x: None for x in ["proxy", "qed", "synth", "score", "smiles"]} for _ in range(len(mols))]

        mol_mol = [mol.mol for mol in mols]
        good = [mmm is not None for mmm in mol_mol]

        for ix, ggg in enumerate(good):
            if ggg:
                infos[ix]["smiles"] = mols[ix].smiles
                infos[ix]["qed"] = qed_score = QED.qed(mol_mol[ix])
                if qed_score <= self.qed_th:
                    good[ix] = False

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
                        infos[ix]["mol"] = mols[ix].dump()

        return res_scores, infos
