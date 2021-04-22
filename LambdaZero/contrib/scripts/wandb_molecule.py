import os.path as osp
import LambdaZero.utils
import wandb

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

logger_config = {
    "wandb": {
        "project": "some_plots",
        #"api_key_file": osp.join(summaries_dir, "wandb_key")
    }}

wandb.init(**logger_config["wandb"])


import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from LambdaZero.contrib.environments.block_mol_graph_v2 import BlockMolGraph_v2
import ray
ray.init()

env = BlockMolGraph_v2({})

obs = env.reset()
trajs = []
for i in range(50):
    #print("num blocks", env.molMDP.molecule.numblocks)
    done = False
    traj = []
    while not done:
        if sum(obs["action_mask"])==1:
            action = 0
            print("end on step", len(traj), "num blocks", env.molMDP.molecule.numblocks,
                  "jbonds", len(env.molMDP.molecule.jbonds), "obs actions", sum(env._action_mask()))
        else:
            action = np.random.choice(1+np.where(obs["action_mask"][1:])[0])
        obs, reward, done, info = env.step(action)
        traj.append(env.molMDP.molecule.smiles)
    obs = env.reset()
    trajs.append(traj)


def log_trajectories(trajs, max_steps):
    mols = []
    for i in range(len(trajs)):
        for j in range(max_steps):
            if j < len(trajs[i]):
                mols.append(Chem.MolFromSmiles(trajs[i][j]))
            else:
                mols.append(Chem.MolFromSmiles("H"))
    img = Draw.MolsToGridImage(mols, molsPerRow=max_steps, subImgSize=(250, 250),)
    wandb.log({"traj": wandb.Image(img)})

log_trajectories(trajs, env.max_steps)

# todo: I need to decide which trajectories to log and how
#wandb.log({"protein": wandb.Molecule(open("/home/maksym/Downloads/4jnc.pdb"))})
#molecule = "/home/maksym/Summaries/docking/docked/5F52PRH6C5ACVW1.pdb"
#wandb.log({"molecule":wandb.Molecule("1")})