import os.path as osp
import LambdaZero.utils
import wandb
import pandas as pd

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

# blocks_file =osp.join(datasets_dir, "fragdb", 'pdb_blocks_55_manualFix2.json')
# #print(osp.exists(blocks_file))
# blocks = pd.read_json(blocks_file)
# print(blocks.to_string())

logger_config = {
    "wandb": {
        "project": "some_plots",
        #"api_key_file": osp.join(summaries_dir, "wandb_key")
    }}





import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from LambdaZero.contrib.environments.block_mol_graph_v2 import BlockMolGraph_v2
import ray
#ray.init()

# env = BlockMolGraph_v2({})
#
# obs = env.reset()
# trajs = []
# for i in range(50):
#     #print("num blocks", env.molMDP.molecule.numblocks)
#     done = False
#     traj = []
#     while not done:
#         if sum(obs["action_mask"])==1:
#             action = 0
#             print("end on step", len(traj), "num blocks", env.molMDP.molecule.numblocks,
#                   "jbonds", len(env.molMDP.molecule.jbonds), "obs actions", sum(env._action_mask()))
#         else:
#             action = np.random.choice(1+np.where(obs["action_mask"][1:])[0])
#         obs, reward, done, info = env.step(action)
#         traj.append(env.molMDP.molecule.smiles)
#     obs = env.reset()
#     trajs.append(traj)


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

#log_trajectories(trajs, env.max_steps)


from PIL import Image
imarray = np.random.rand(100,100,3) * 255
im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')

wandb.init()
wandb.log({"false_traj2": wandb.Image(im)})











# todo: I need to decide which trajectories to log and how
# env: episode_smiles [ CC, CO, ..]
# reward -> episode_smiles
# proxy: before_acquire: log_best_trajectories, log_worst_trajectories

# todo: I want to monitor molecules being acquired
# proxy_after_acquire: log_best_trajectories

# todo: after_acquire: log table
# smiles, synth, qed, dockscore, step
# ideally I would maintain 50 best molecules/run in a table
# if new in top_50 add_to_table

# todo: after_acquire_log 3D structures
# I think in this case I would take smiles + log coordinates
# ideally I would log mol2molblock


#wandb.log({"protein": wandb.Molecule(open("/home/maksym/Downloads/4jnc.pdb"))})
#molecule = "/home/maksym/Summaries/docking/docked/5F52PRH6C5ACVW1.pdb"
#wandb.log({"molecule":wandb.Molecule("1")})