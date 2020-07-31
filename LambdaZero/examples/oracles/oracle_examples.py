import time
import ray
import numpy as np
import pandas as pd
#import LambdaZero
from rdkit.Chem import QED
import os.path as osp
from copy import deepcopy
from rdkit import Chem
from matplotlib import pyplot as plt

import LambdaZero.chem
import LambdaZero.utils
import LambdaZero.environments
from LambdaZero.environments.block_mol_v3 import DEFAULT_CONFIG as block_mol_v3_config

class cfg:
    datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
    nsamples = 2000
    dock_nsamples = 128
    num_cpus = 8

    #db_name = "run_name"
    #db_path = osp.join(datasets_dir, db_name)
    #results_dir = osp.join(datasets_dir, db_name, "results")
    out_dir = osp.join(summaries_dir, "oracle_examples")

    # env parameters
    blocks_file = osp.join(datasets_dir,"fragdb/blocks_PDB_105.json")

    # reward config
    block_mol_v3_reward_config = block_mol_v3_config["reward_config"]
    dockscore_norm = [-43.042, 7.057]

    # MPNN parameters
    #dockscore_model = osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002")

    # docking parameters
    dock6_dir = osp.join(programs_dir, "dock6")
    chimera_dir = osp.join(programs_dir, "chimera")
    docksetup_dir = osp.join(datasets_dir, "brutal_dock/d4/docksetup")



# example: docking a molecule to virus (COVID19protein
dock_smi = LambdaZero.chem.Dock_smi(outpath=cfg.out_dir,
                          chimera_dir=cfg.chimera_dir,
                          dock6_dir=cfg.dock6_dir,
                          docksetup_dir= osp.join(cfg.datasets_dir, "brutal_dock/mpro_6lze/docksetup"),
                          gas_charge=True)
# smi = "[O-]C(=O)[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H](CC2CCCCC2)NC(=O)c3[nH]c4ccccc4c3"
# name, energy, coord = dock_smi.dock(smi)
# print("dock energy should be -50 to -60:", energy)


# generate random molecules
molecules = []
molMDP = LambdaZero.environments.molMDP.MolMDP(blocks_file=cfg.blocks_file)
for i in range(cfg.nsamples):
    molMDP.reset()
    molMDP.random_walk(5)
    molecules.append(molMDP.molecule)
    #print(Chem.MolToSmiles(molMDP.molecule.mol))

# evaluate random molecules with predicted dock reward
rewards = []
predDockRevard_v2 = LambdaZero.environments.reward.PredDockReward_v2(**cfg.block_mol_v3_reward_config)
for molecule in molecules:
    predDockRevard_v2.reset()
    reward = predDockRevard_v2(molecule, env_stop=False, simulate=True, num_steps=1)
    reward = pd.DataFrame({k:[v] for k,v in reward[1].items()})
    reward["smiles"] = [Chem.MolToSmiles(molecule.mol)]
    rewards.append(reward)
rewards = pd.concat(rewards, axis=0, ignore_index=True)
print(rewards)

# build correlation plot between reward and dockscore
synth_mask = rewards["synth"].to_numpy() > 4
qed_mask = rewards["qed"].to_numpy() > 0.5
good_sample_idx = np.where(np.logical_and(synth_mask,qed_mask))[0][:cfg.dock_nsamples]
bad_sample_idx = np.where(np.logical_and(~synth_mask,~qed_mask))[0][:cfg.dock_nsamples]


@ray.remote
def _dock_smi(smi):
    try:
        gridscore = dock_smi.dock(smi)[1]
        dock_reward = -((gridscore - cfg.dockscore_norm[0]) / cfg.dockscore_norm[1])
    except Exception as e:
        dock_reward = None
    return dock_reward

ray.init(num_cpus=cfg.num_cpus)
smis = rewards["smiles"].to_numpy()[good_sample_idx]
good_dockrewards = ray.get([_dock_smi.remote(smi) for smi in smis])
smis = rewards["smiles"].to_numpy()[bad_sample_idx]
bad_dockrewards = ray.get([_dock_smi.remote(smi) for smi in smis])

dock_rewards = rewards["dock_reward"].to_numpy()

plt.figure(dpi=300)
plt.xlabel("mpnn_dock_reward")
plt.ylabel("dock_reward")
plt.scatter(dock_rewards[good_sample_idx], good_dockrewards, label="synth > 5 & qed > 0.5")
plt.scatter(dock_rewards[bad_sample_idx], bad_dockrewards, label="synth < 5 & qed < 0.5")
plt.gca().legend()
plt.savefig(osp.join(cfg.out_dir,"DockRevard_v3_sample.png"))

good_dockrewards_mae = np.abs(dock_rewards[good_sample_idx] - good_dockrewards).mean()
bad_dockrewards_mae = np.abs(dock_rewards[bad_sample_idx] - bad_dockrewards).mean()
print("good/bad dockrewards mae", good_dockrewards_mae, bad_dockrewards_mae)


# # # evaluate random molecules with predicted dock reward
# reward = LambdaZero.environments.reward.PredDockReward(load_model=cfg.dockscore_model,
#                         natm_cutoff=[45, 50],
#                         qed_cutoff=[0.2, 0.7],
#                         soft_stop=False,
#                         exp=None,
#                         delta=False,
#                         simulation_cost=0.0,
#                         device="cuda")
# reward.reset()
# print("predicted reward:", reward(molMDP.molecule,env_stop=False,simulate=True,num_steps=1))
#
#
# # # evaluate random molecules with docking
# dock_smi = LambdaZero.chem.Dock_smi(outpath=cfg.out_dir,
#                          chimera_dir=cfg.chimera_dir,
#                          dock6_dir=cfg.dock6_dir,
#                          docksetup_dir=cfg.docksetup_dir)
#
# name, energy, coord = dock_smi.dock(Chem.MolToSmiles(molMDP.molecule.mol))
# print("dock energy:", energy)




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