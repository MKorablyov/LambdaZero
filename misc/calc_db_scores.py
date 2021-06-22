import cv2
import gym
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os.path as osp
import torch
import json
import os

from LambdaZero.contrib.oracle.oracle import PreDockingDB
from lightrl import env
from lightrl.env.reward import DummyReward
from lightrl.env.scores import QEDEstimator, ParallelSynth
from lightrl.env.molMDP import MolMDP
from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def calc_db_scores(use_cuda=True):
    log_freq = 1000
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Save path
    save_scores = os.path.join(datasets_dir, "dock_db_scores")
    if not os.path.isdir(save_scores):
        os.mkdir(save_scores)

    # Load mol Converter
    mol_mdp = MolMDP(osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"))

    # Load scoring classes
    synth_net = ParallelSynth(use_cuda=use_cuda)
    synth_net.to(device)
    synth_net.share_memory()
    qed = QEDEstimator()

    # Load Database
    db_name, _ = PreDockingDB.get_last_db()
    score_path = PreDockingDB.get_db_scores_path(db_name)

    df = PreDockingDB.read_last_db()

    # Calculate scores
    res_scores = []
    for ismi, (smi, state) in enumerate(df.iterrows()):
        qed_score = qed.eval(smi)
        mol = mol_mdp.load(state)
        synth_score = synth_net([mol.mol])[0]
        res_scores.append([smi, qed_score, synth_score])

        if (ismi + 1) % log_freq == 0:
            print(f"Done {ismi}/{len(df)}")

    res_scores = pd.DataFrame(res_scores, columns=["smiles", "qed_score", "synth_score"])
    res_scores.to_pickle(score_path)
    return res_scores

if __name__ == "__main__":
    res_scores = calc_db_scores()