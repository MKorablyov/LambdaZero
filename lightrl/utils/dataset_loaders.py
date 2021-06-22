import cv2
import sys
import gym
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
from argparse import Namespace
import itertools
import time
import torch
import copy
from tqdm import tqdm

from LambdaZero.contrib.oracle.oracle import PreDockingDB
from lightrl import env
from lightrl.env.reward import DummyReward
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

def pool_get_mol_graph(largs):
    state, env_molMDP, env_graph_mol_obs, transform = largs
    mol = env_molMDP.load(state)
    graph, flat_graph = env_graph_mol_obs(mol, flatten=False)
    if transform is not None:
        graph = transform(graph)

    traj = []
    traj_mol = mol

    while True:
        parents = env_molMDP.parents(traj_mol)
        parent_idx = np.random.randint(len(parents))
        parent = list(parents[parent_idx])
        traj.append(parent)
        traj_mol = parent[0]
        numblocks = traj_mol.numblocks
        parent[0] = env_graph_mol_obs(traj_mol, flatten=False)[0]
        if numblocks <= 1:
            break

    return graph, traj


def load_predocked_dataset_with_children(args: Namespace):
    # Get config
    num_pool = getattr(args, "mol_graph_pool", 20)
    sample_size = getattr(args, "sample_size", 0)
    filter_by_score = getattr(args, "filter_candidates", False)
    with_scores = getattr(args, "with_scores", False) or filter_by_score
    qed_th = getattr(args, "qed_th", 0.3)
    synth_th = getattr(args, "synth_th", 4.)

    gather_sample_size = sample_size
    if filter_by_score:
        gather_sample_size = 0

    # Get predocked db
    df = PreDockingDB.read_last_db(sample_size=gather_sample_size, with_scores=with_scores)
    orig_set_len = len(df)
    if filter_by_score:
        df = df[(df["qed_score"] >= qed_th) & (df["synth_score"] >= synth_th)]
        if sample_size != 0:
            df = df.iloc[:sample_size]

    # Print some stats
    dock_hist = list(zip(*np.histogram(df.dockscore.values, bins=100)))
    print("Dock 100 histogram\n", dock_hist)

    print(f"Loaded after filtering {len(df)} (/{orig_set_len}) states with {len(np.unique(df.index))} unique smiles")

    # Load Env to have methods for loading molMDP from state and converting them to graphs
    env = gym.make("BlockMolEnvGraph-v1", config={
        "obs_cuda": False,
    }, proc_id=0)
    transform = None
    env.molMDP.build_translation_table()

    st = time.time()
    all_graph_traj = []
    state_records = df.to_dict(orient='records')

    print(f"Starting pool with {num_pool} children for {len(state_records)} records")
    with torch.multiprocessing.Pool(num_pool) as p:
        bsize = 1000
        batch_idxs = range(0, len(state_records), bsize)

        for ibtch, batch in tqdm(enumerate(batch_idxs), total=len(batch_idxs)):
            graphs_with_traj = p.map(
                pool_get_mol_graph,
                zip(state_records[batch: batch + bsize],
                    itertools.repeat(env.molMDP),
                    itertools.repeat(env.graph_mol_obs),
                    itertools.repeat(transform))
            )

            # Free mem from shared
            clone_g = [copy.deepcopy(x) for x in graphs_with_traj]
            for x in graphs_with_traj:
                del x
            del graphs_with_traj
            all_graph_traj += clone_g

    """
        # Serial solution In case of problems with parallel version
    for ibtch in range(len(state_records)):
        all_graph_traj.append(
            pool_get_mol_graph((state_records[ibtch], env.molMDP, env.graph_mol_obs, transform))
        )
        if (ibtch + 1) % 1000 == 0:
            print(f"DONE : {len(all_graph_traj)}")
    """

    print(f"Finish loading in {time.time() - st}")
    trajectories = []
    for (g, traj), y in zip(all_graph_traj, df.dockscore.values):
        traj = traj[::-1]
        traj.append([g, (None, None)])
        trajectories.append((traj, y))

    # for iii, (traj, dockscore, traj_mol) in enumerate(trajectories):
    #     env.reset()
    #     env.molMDP.reset()
    #     env.molMDP.molecule = traj_mol
    #     env.num_steps = 0
    #
    #     _obs = env._make_obs()[0]
    #     print(
    #         traj_mol.blockidxs,
    #         traj_mol.slices,
    #         traj_mol.jbonds,
    #         traj_mol.stems,
    #     )
    #     print(
    #         df.to_dict(orient='records')[iii]
    #     )
    #     # print((_obs["mol_graph"].x.cpu() == traj[0][0].x).all(), (_obs["mol_graph"].edge_attr.cpu() == traj[0][0].edge_attr).all())
    #     for itr, (mol, (block_idx, stem_idx)) in enumerate(traj[:-1]):
    #         print(itr, block_idx, stem_idx)
    #         obs, _, _, _  = env.step(105 * stem_idx + block_idx + 7)
    #         x, y = obs["mol_graph"].x.shape == traj[itr+1][0].x.shape, obs["mol_graph"].edge_attr.shape == traj[itr+1][0].edge_attr.shape
    #         print(itr, x, y)
    #         assert x and y, "Not the ssame"
    # import pdb;
    # pdb.set_trace()

    return trajectories


if __name__ == "__main__":
    args = Namespace(sample_size=0, filter_candidates=True)
    traj = load_predocked_dataset_with_children(args)
    print(len(traj))