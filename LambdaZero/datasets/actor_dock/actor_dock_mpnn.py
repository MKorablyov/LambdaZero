import time, os.path as osp
from copy import deepcopy
from rdkit import Chem
import numpy as np
import LambdaZero.utils
import LambdaZero.environments

class cfg:
    datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
    num_cpus = 8

    db_name = "actor_dock"
    db_path = osp.join(datasets_dir, db_name)
    results_dir = osp.join(datasets_dir, db_name, "results")
    out_dir = osp.join(datasets_dir, db_name, "dock")

    # env parameters
    blocks_file = osp.join(datasets_dir,"fragdb/blocks_PDB_105.json")

    # MPNN parameters
    dockscore_model = osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002")


# @ray.remote
# class Worker():
# def __init__()
#   molMDP
#   rewars
# def sample()
#   return energies_for_molecule

# give 10 jobs per worker
# tasks = [[worker.sample() for worker in 100] for 10]

# while True:
#   # wait for 1000 jobs to complete and restart jobs
#   ray_wait_for(1000)
#   ray_restart_finished()
#   save_resuts


#initialize MDP
molMDP = LambdaZero.environments.molMDP.MolMDP(blocks_file=cfg.blocks_file)
# Initialize dockign reward prediction
comp_reward = LambdaZero.environments.reward.PredDockReward(load_model=cfg.dockscore_model,
                        natm_cutoff=[45, 50],
                        qed_cutoff=[0.2, 0.7],
                        soft_stop=False,
                        exp=None,
                        delta=False,
                        simulation_cost=0.0,
                        device="cuda")


# number of possible substitutes = num_building_blocks (105) * number_of_places_to_modify (usually about 5-10)
# we want to be able to do 1st


def compute_action_values(molMDP):
    init_molecule = deepcopy(molMDP.molecule)
    # iterate over places to change
    action_values = []
    for atmidx in molMDP.molecule.stem_atmidxs:
        # iterate over alphabet of building blocks
        addblock_values = []
        for block_idx in range(molMDP.num_blocks):
            molMDP.add_block(block_idx=block_idx, atmidx=atmidx)
            reward = comp_reward(molMDP.molecule, env_stop=False,simulate=True,num_steps=1)[0]
            addblock_values.append(reward)
            molMDP.molecule = deepcopy(init_molecule)
        action_values.append(addblock_values)
    return np.asarray(action_values)


# iterate throuch a bunch of molecules
for i in range(100):
    molMDP.reset()
    comp_reward.reset()
    molMDP.random_walk(5)
    action_values = compute_action_values(molMDP=molMDP)

    print("action_values", action_values.shape)


# molMDP().random_walk() -> mol_graph, r-groups
# choose r-group randomly
# molMDP().make_subs(r-group) -> mol_graph * 105
# MPNN(mol_graph, r_group) -> logit * 105           # actor
# MPNN(mol_graphs) -> label * 105                   # critic