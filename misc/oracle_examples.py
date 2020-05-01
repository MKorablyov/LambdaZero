import time
import ray
import numpy as np
import pandas as pd
#import LambdaZero
from rdkit.Chem import QED
import os.path as osp
from LambdaZero.environments.molecule import MolMDP
from LambdaZero import chem
from LambdaZero.environments.molecule import PredDockReward

from rdkit import Chem

class cfg:

    datasets_dir = osp.join("/home/maksym", "Datasets")
    programs_dir = osp.join("/home/maksym/", "Programs")
    num_cpus = 8

    db_name = "actor_dock"
    db_path = osp.join(datasets_dir, db_name)
    results_dir = osp.join(datasets_dir, db_name, "results")
    out_dir = osp.join(datasets_dir, db_name, "dock")

    # MPNN parameters
    dockscore_model = osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002")

    # docking parameters
    dock6_dir = osp.join(programs_dir, "dock6")
    chimera_dir = osp.join(programs_dir, "chimera")
    docksetup_dir = osp.join(datasets_dir, "brutal_dock/d4/docksetup")


# generate random molecules
molMDP = MolMDP(blocks_file="/home/maksym/Datasets/fragdb/blocks_PDB_105.json")
for i in range(10):
    molMDP.reset()
    molMDP.random_walk(5)
    print(Chem.MolToSmiles(molMDP.molecule.mol))


# evaluate random molecules with predicted dock reward
reward = PredDockReward(load_model=cfg.dockscore_model,
                        natm_cutoff=[45, 50],
                        qed_cutoff=[0.2, 0.7],
                        soft_stop=False,
                        exp=None,
                        delta=False,
                        simulation_cost=0.0,
                        device="cuda")
reward.reset()
print("predicted reward:", reward(molMDP.molecule,env_stop=False,simulate=True,num_steps=1))


# evaluate random molecules with docking
dock_smi = chem.Dock_smi(outpath=cfg.out_dir,
                         chimera_dir=cfg.chimera_dir,
                         dock6_dir=cfg.dock6_dir,
                         docksetup_dir=cfg.docksetup_dir)

name, energy, coord = dock_smi.dock(Chem.MolToSmiles(molMDP.molecule.mol))
print("dock energy:", energy)




# ray remote function
# @ray.remote
# def func(args):
#     sum(args)
#     time.sleep(np.abs(np.mean(sum(args))))
#
#     return np.mean(args), np.var(args)
#
#
# if __name__ == "__main__":
#     # ray.init()
#     #
#     # futures = []
#     # for i in range(10):
#     #     mol = np.random.uniform(-0.1,0.1,size=10)
#     #     futures.append(func.remote(mol))
#     #     print(i)
#     #
#     # df = ray.get(futures)
#     # df = pd.concat([pd.DataFrame({"mean":[o[0]], "variance":[o[1]]}) for o in df])
#     # df.reset_index(inplace=True)
#     # df.to_feather("/home/maksym/Desktop/stuff.feather")
#
#     df = pd.read_feather("/home/maksym/Desktop/stuff.feather")
#     print(df)