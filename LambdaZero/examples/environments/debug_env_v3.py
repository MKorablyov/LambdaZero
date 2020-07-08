import os.path as osp
import numpy as np
from LambdaZero.environments.block_mol_v3 import BlockMolEnv_v3, DEFAULT_CONFIG
import LambdaZero.utils

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

config = DEFAULT_CONFIG
config["obs_config"]["mol_fp_len"] = 4
config["obs_config"]["stem_fp_len"] = 4
config["molMDP_config"]["blocks_file"] = osp.join(datasets_dir, "fragdb/pdb_blocks_330.json")
env = BlockMolEnv_v3(config)




obs = env.reset()
for i in range(10000):

    action = np.random.choice(np.where(obs["action_mask"])[0])
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

    print("step:", i, done, "done")
