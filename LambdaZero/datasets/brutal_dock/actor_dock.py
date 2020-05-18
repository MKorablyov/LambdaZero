import time, os.path as osp
import numpy as np
from LambdaZero.environments.molecule import BlockMolEnv_v3
import ray
import os
from LambdaZero.examples.AlphaZero.config import mol_blocks_v4_config
from rdkit import Chem
from LambdaZero.chem import Dock_smi
import pandas as pd

def exhaustive_subs(env, env_config):
    # find some starting state
    max_blocks = env_config["max_blocks"]
    num_blocks = env_config["num_blocks"]
    env = env(env_config)
    done = False
    while not done:
        obs = env.reset()
        stem_mask = obs["action_mask"][max_blocks:]
        if sum(stem_mask) > 0:
            done = True
            num_stems = (1 + np.argmax(np.where(stem_mask)[0])) // num_blocks

    # make smiles array
    istate = env.get_state()
    smis = [Chem.MolToSmiles(env.molMDP.molecule.mol)]
    for action_idx in range(num_stems * num_blocks):
        env.step(max_blocks + action_idx)
        smis.append(Chem.MolToSmiles(env.molMDP.molecule.mol))
        env.set_state(istate)
    return smis


class Empty:
    def __init__(self, **kwargs):
        pass

    def reset(self):
        pass

    def __call__(self, *args, **kwargs):
        return None, {}

class cfg:
    # temp
    ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    datasets_dir = osp.join(ROOT, "Datasets")
    programs_dir = osp.join(ROOT, "Programs")
    num_cpus = 8

    db_name = "actor_dock"
    db_path = osp.join(datasets_dir, db_name)
    results_dir = osp.join(datasets_dir, db_name, "results")
    dock_dir = osp.join(datasets_dir, db_name, "dock")

    # env
    env_config = mol_blocks_v4_config()["env_config"]
    env_config["random_steps"] = 3
    env_config["reward"] = Empty
    env_config["obs"] = Empty

    # docking parameters
    dock6_dir = osp.join(programs_dir, "dock6")
    chimera_dir = osp.join(programs_dir, "chimera")
    docksetup_dir = osp.join(datasets_dir, "brutal_dock/d4/docksetup")


if __name__ == "__main__":
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)
    #ray.init(memory=8*10**9, num_cpus=cfg.num_cpus)
    ray.init(address='auto')

    dock_smi = Dock_smi(outpath=cfg.dock_dir,
                        chimera_dir=cfg.chimera_dir,
                        dock6_dir=cfg.dock6_dir,
                        docksetup_dir=cfg.docksetup_dir)

    @ray.remote
    def ray_dock_smi(smi):
        try:
            name, gridscore, coord = dock_smi.dock(smi)
            coord = np.asarray(coord,dtype=np.float32).tolist()
        except Exception as e:
            print(e)
            name, gridscore, coord = None, None, None
        df = pd.DataFrame({"smi": [smi], "name": [name], "gridscore": [gridscore], "coord":[coord]})
        return df

    while True:
        smis = exhaustive_subs(BlockMolEnv_v3, cfg.env_config)
        tasks = [ray_dock_smi.remote(smi) for smi in smis]
        start = time.time()
        docked = ray.get(tasks)
        docked = pd.concat(docked, ignore_index=True)
        df_out = os.path.join(cfg.results_dir, smis[0] + ".parquet")
        docked.to_parquet(df_out, engine="fastparquet", compression="UNCOMPRESSED")

        # print(pd.read_parquet(df_out, engine="fastparquet"))
        # print("exps:", "%.3f" % (64. / (time.time() - start)))
        # # test function
        # files = os.listdir(cfg.results_dir)
        # for f in files:
        #     print(pd.read_parquet(os.path.join(cfg.results_dir, f), engine="fastparquet")["gridscore"])
