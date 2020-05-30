import numpy as np

from .block_mol_v4 import BlockMolEnv_v4

class BlockMolEnv_v5(BlockMolEnv_v4):
    def __init__(self, config=None):
        BlockMolEnv_v4.__init__(self, config)
        self.env_reset = super().reset
        self.env_step = super().step

        self.buff_len = 1000
        self.fp_buff = []
        self.buff = []

    def reset(self):
        return self.env_reset()

    def step(self, action):
        obs, reward, done, info = self.env_step(action)

        if done:
            mol_fp = obs["mol_fp"]

            if len(self.fp_buff) > 1:
                fp_buff = np.asarray(self.fp_buff)
                dist = np.sum(np.abs(fp_buff - mol_fp[None, :]), axis=1)
                dist = 1 - (dist / (np.sum(np.abs(fp_buff),1) + np.sum(np.abs(mol_fp[None,:]),1)))
                print("min dist", np.max(dist), "mean dist",  np.mean(dist))

            self.fp_buff.append(mol_fp)
            if len(self.fp_buff) > self.buff_len: self.fp_buff.pop(0)

            #mol = self.MolMDP.molecule.mol
        #if done and action == 0:
            #mol_attr, num_steps, num_simulations, previous_reward, mol = self.get_state()
            #print("action", action, "num_steps", self.num_steps)
        return obs, reward, done, info
