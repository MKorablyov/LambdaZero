import time
from copy import deepcopy
import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Dict, Box
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .. import chem


class BlockMoleculeData:
    # todo: make properties fast
    def __init__(self):
        self.blockidxs = []       # indexes of every block
        self.blocks = []          # rdkit molecule objects for every
        self.slices = [0]         # atom index at which every block starts
        self.numblocks = 0
        self.jbonds = []          # [block1, block2, bond1, bond2]
        self.stems = []           # [block1, bond1]
        self._mol = None

    def add_block(self, block_idx, block, block_r, stem_idx, atmidx):
        self.blockidxs.append(block_idx)
        self.blocks.append(block)
        self.slices.append(self.slices[-1] + block.GetNumAtoms())
        self.numblocks += 1
        [self.stems.append([self.numblocks-1,r]) for r in block_r[1:]]

        if len(self.blocks)==1:
            self.stems.append([self.numblocks-1, block_r[0]])
        else:
            if stem_idx is None:
                assert atmidx is not None, "need stem or atom idx"
                stem_idx = np.where(self.stem_atmidxs==atmidx)[0][0]
            else:
                assert atmidx is None, "can't use stem and atom indices at the same time"

            stem = self.stems[stem_idx]
            bond = [stem[0], self.numblocks-1, stem[1], block_r[0]]
            self.stems.pop(stem_idx)
            self.jbonds.append(bond)
            # destroy properties
            self._mol = None
        return None

    def delete_blocks(self, block_mask):
        # update number of blocks
        self.numblocks = np.sum(np.asarray(block_mask, dtype=np.int32))
        self.blocks = list(np.asarray(self.blocks)[block_mask])
        self.blockidxs = list(np.asarray(self.blockidxs)[block_mask])

        # update junction bonds
        reindex = np.cumsum(np.asarray(block_mask,np.int32)) - 1
        jbonds = []
        for bond in self.jbonds:
            if block_mask[bond[0]] and block_mask[bond[1]]:
                jbonds.append(np.array([reindex[bond[0]], reindex[bond[1]], bond[2], bond[3]]))
        self.jbonds = jbonds

        # update r-groups
        stems = []
        for stem in self.stems:
            if block_mask[stem[0]]:
                stems.append(np.array([reindex[stem[0]],stem[1]]))
        self.stems = stems

        # update slices
        natms = [block.GetNumAtoms() for block in self.blocks]
        self.slices = [0] + list(np.cumsum(natms))

        # destroy properties
        self._mol = None
        return reindex

    def remove_jbond(self, jbond_idx=None, atmidx=None):

        if jbond_idx is None:
            assert atmidx is not None, "need jbond or atom idx"
            jbond_idx = np.where(self.jbond_atmidxs == atmidx)[0][0]
        else:
            assert atmidx is None, "can't use stem and atom indices at the same time"

        # find index of the junction bond to remove
        jbond = self.jbonds.pop(jbond_idx)

        # find the largest connected component; delete rest
        jbonds = np.asarray(self.jbonds, dtype=np.int32)
        jbonds = jbonds.reshape([len(self.jbonds),4]) # handle the case when single last jbond was deleted
        graph = csr_matrix((np.ones(self.numblocks-2),
                            (jbonds[:,0], jbonds[:,1])),
                           shape=(self.numblocks, self.numblocks))
        _, components = connected_components(csgraph=graph, directed=False, return_labels=True)
        block_mask = components==np.argmax(np.bincount(components))
        reindex = self.delete_blocks(block_mask)

        if block_mask[jbond[0]]:
            stem = np.asarray([reindex[jbond[0]], jbond[2]])
        else:
            stem = np.asarray([reindex[jbond[1]], jbond[3]])
        self.stems.append(stem)
        atmidx = self.slices[stem[0]] + stem[1]
        return atmidx

    @property
    def stem_atmidxs(self):
        stems = np.asarray(self.stems)
        if stems.shape[0]==0:
            stem_atmidxs = np.array([])
        else:
            stem_atmidxs = np.asarray(self.slices)[stems[:,0]] + stems[:,1]
        return stem_atmidxs

    @property
    def jbond_atmidxs(self):
        jbonds = np.asarray(self.jbonds)
        if jbonds.shape[0]==0:
            jbond_atmidxs = np.array([])
        else:
            jbond_atmidxs = np.stack([np.concatenate([np.asarray(self.slices)[jbonds[:,0]] + jbonds[:,2]]),
                                      np.concatenate([np.asarray(self.slices)[jbonds[:,1]] + jbonds[:,3]])],1)
        return jbond_atmidxs

    @property
    def mol(self):
        if self._mol == None:
            self._mol, _ = chem.mol_from_frag(jun_bonds=self.jbonds, frags=self.blocks)
        return self._mol


class MolMDP:
    def __init__(self, blocks_file):
        blocks = pd.read_json(blocks_file)
        self.block_smi = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()
        self.block_nrs = np.asarray([len(r) for r in self.block_rs])
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in blocks["block_smi"]]
        self.block_natm = np.asarray([b.GetNumAtoms() for b in self.block_mols])
        self.num_blocks = len(self.block_smi)
        self.reset()

    def reset(self):
        self.molecule = BlockMoleculeData()
        return None

    def add_block(self, block_idx, stem_idx=None, atmidx=None):
        assert (block_idx >= 0) and (block_idx <= len(self.block_mols)), "unknown block"
        self.molecule.add_block(block_idx,
                                block=self.block_mols[block_idx],
                                block_r=self.block_rs[block_idx],
                                stem_idx=stem_idx, atmidx=atmidx)
        return None

    def remove_jbond(self, jbond_idx=None, atmidx=None):
        atmidx = self.molecule.remove_jbond(jbond_idx, atmidx)
        return atmidx

    def random_walk(self, length):
        done = False
        while not done:
            if self.molecule.numblocks==0:
                block_idx = np.random.choice(np.arange(self.num_blocks))
                stem_idx = None
                self.add_block(block_idx=block_idx, stem_idx=stem_idx)
            elif len(self.molecule.stems) > 0:
                block_idx = np.random.choice(np.arange(self.num_blocks))
                stem_idx = np.random.choice(len(self.molecule.stems))
                self.add_block(block_idx=block_idx, stem_idx=stem_idx)
                if self.molecule.numblocks >= length: done = True
            else:
                self.reset()


class FPEmbedding_v2:
    def __init__(self, mol_fp_len, mol_fp_radiis, stem_fp_len, stem_fp_radiis):
        self.mol_fp_len = mol_fp_len
        self.mol_fp_radiis = mol_fp_radiis
        self.stem_fp_len = stem_fp_len
        self.stem_fp_radiis = stem_fp_radiis

    def __call__(self, molecule):
        mol = molecule.mol
        mol_fp = chem.get_fp(mol, self.mol_fp_len, self.mol_fp_radiis)

        # get fingerprints and also handle empty case
        stem_fps = [chem.get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx]) for idx in molecule.stem_atmidxs]

        jbond_fps = [(chem.get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx[0]]) +
                     chem.get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx[1]]))/2.
                     for idx in molecule.jbond_atmidxs]

        if len(stem_fps) > 0: stem_fps = np.stack(stem_fps, 0)
        else: stem_fps = np.empty(shape=[0, self.stem_fp_len * len(self.stem_fp_radiis)],dtype=np.float32)
        if len(jbond_fps) > 0: jbond_fps = np.stack(jbond_fps, 0)
        else: jbond_fps = np.empty(shape=[0, self.stem_fp_len * len(self.stem_fp_radiis)],dtype=np.float32)

        return mol_fp, stem_fps, jbond_fps

class BlockMolEnv_v3:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]
    # sample termination (0 reward if not terminated)

    def __init__(self, config=None):
        self.num_blocks = config["num_blocks"]
        self.max_steps = config["max_steps"]
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.random_steps = config["random_steps"]
        self.allow_removal = config["allow_removal"]
        num_actions = self.max_blocks + self.max_branches * self.num_blocks

        self.action_space = Discrete(num_actions,)
        mol_fp_len = config["obs_config"]["mol_fp_len"] * len(config["obs_config"]["mol_fp_radiis"])
        stem_fp_len = config["obs_config"]["stem_fp_len"] * len(config["obs_config"]["stem_fp_radiis"])
        self.observation_space = Dict({
            "mol_fp": Box(low=0, high=1, shape=(mol_fp_len,)),
            "stem_fps": Box(low=0, high=1, shape=(self.max_branches, stem_fp_len,)),
            "jbond_fps": Box(low=0, high=1, shape=(self.max_blocks -1, stem_fp_len,)),
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
        })

        self.molMDP = MolMDP(**config["molMDP_config"])
        self.reward = config["reward"](**config["reward_config"])
        self.get_fps = FPEmbedding_v2(**config["obs_config"])

    def _make_obs(self):
        mol_fp, stem_fps_, jbond_fps_ = self.get_fps(self.molMDP.molecule)

        # pad indices
        stem_fps = np.zeros([self.max_branches, stem_fps_.shape[1]], dtype=np.float32)
        stem_fps[:stem_fps_.shape[0], :] = stem_fps_[:self.max_branches,:]
        jbond_fps = np.zeros([self.max_blocks -1, stem_fps_.shape[1]], dtype=np.float32)
        jbond_fps[:jbond_fps_.shape[0], :] = jbond_fps_[:self.max_blocks-1, :]

        # make action mask
        jbond_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        if self.allow_removal:
            jbond_mask[:jbond_fps_.shape[0]] = 1
        stem_mask = np.zeros(self.max_branches, dtype=np.float32)
        if self.molMDP.molecule.numblocks == 0: stem_mask[0] = 1 # allow to add any block
        else: stem_mask[:stem_fps_.shape[0]] = 1

        stem_mask = np.tile(stem_mask[:, None], [1, self.num_blocks]).reshape(-1)
        action_mask = np.concatenate([np.ones([1], dtype=np.float32), jbond_mask, stem_mask])

        obs = {"mol_fp": mol_fp,
               "stem_fps": stem_fps,
               "jbond_fps": jbond_fps,
               "action_mask": action_mask,
               "num_steps": self.num_steps}

        return obs

    def _if_terminate(self):
        terminate = False
        molecule = self.molMDP.molecule
        # max steps
        if self.num_steps >= self.max_steps: terminate = True
        # max_branches
        if len(molecule.stems) >= self.max_branches: terminate = True
        # max blocks
        if len(molecule.jbond_atmidxs) >= self.max_blocks-1: terminate = True
        # max_atoms
        if molecule.slices[-1] >= self.max_atoms: terminate = True
        return terminate

    def reset(self):
        self.num_steps = 0
        self.molMDP.reset()
        self.reward.reset()
        obs = self._make_obs()
        for i in range(self.random_steps):
            actions = np.where(obs["action_mask"])[0]
            action = np.random.choice(actions)
            self.step(action)
            obs = self._make_obs()
            if self._if_terminate():
                self.num_steps = 0
                self.molMDP.reset()
        self.num_steps = 0
        return obs

    def step(self, action):
        if (action == 0):
            agent_stop = True
        elif action <= (self.max_blocks - 1):
            agent_stop = False
            self.molMDP.remove_jbond(jbond_idx=action-1)
        else:
            agent_stop = False
            stem_idx = (action - self.max_blocks) // self.num_blocks
            block_idx = (action - self.max_blocks) % self.num_blocks
            self.molMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)

        self.num_steps += 1
        obs = self._make_obs()
        env_stop = self._if_terminate()
        reward, info = self.reward(self.molMDP.molecule, agent_stop, env_stop, self.num_steps)
        info["molecule"] = self.molMDP.molecule
        done = any((agent_stop, env_stop))
        return obs, reward, done, info

    def get_state(self):
        mol_attr = {attr: deepcopy(getattr(self.molMDP.molecule, attr)) for attr in self.mol_attr}
        num_steps = self.num_steps
        return mol_attr, num_steps

    def set_state(self,state):
        mol_attr, self.num_steps = state
        [setattr(self.molMDP.molecule, key, deepcopy(value)) for key, value in mol_attr.items()]
        self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx] for idx in state[0]["blockidxs"]]
        self.molMDP.molecule._mol = None
        return self._make_obs()

    def render(self, outpath):
        mol = self.molMDP.molecule.mol
        if mol is not None: Chem.Draw.MolToFile(mol, outpath)


class FPObs_v1:
    def __init__(self, config, molMDP):
        self.num_blocks = config["num_blocks"]
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.max_steps = config["max_steps"]
        self.molMDP = molMDP

        num_actions = self.max_blocks + self.max_branches * self.num_blocks

        mol_fp_len = config["obs_config"]["mol_fp_len"] * len(config["obs_config"]["mol_fp_radiis"])
        stem_fp_len = config["obs_config"]["stem_fp_len"] * len(config["obs_config"]["stem_fp_radiis"])

        self.action_space = Discrete(num_actions,)
        self.observation_space = Dict({
            "mol_fp": Box(low=0, high=1, shape=(mol_fp_len,)),
            "stem_fps": Box(low=0, high=1, shape=(self.max_branches, stem_fp_len,)),
            "jbond_fps": Box(low=0, high=1, shape=(self.max_blocks-1, stem_fp_len,)),
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
        })
        self.get_fps = FPEmbedding_v2(**config["obs_config"])

    def __call__(self, molecule, num_steps):
        mol_fp, stem_fps_, jbond_fps_ = self.get_fps(molecule)
        # pad indices
        stem_fps = np.zeros([self.max_branches, stem_fps_.shape[1]], dtype=np.float32)
        stem_fps[:stem_fps_.shape[0], :] = stem_fps_[:self.max_branches, :]
        jbond_fps = np.zeros([self.max_blocks - 1, stem_fps_.shape[1]], dtype=np.float32)
        jbond_fps[:jbond_fps_.shape[0], :] = jbond_fps_[:self.max_blocks-1, :]

        # make action mask
        break_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        break_mask[:jbond_fps_.shape[0]] = 1

        # max number of atoms
        atoms_mask = self.molMDP.block_natm <= (self.max_atoms - molecule.slices[-1])
        branches_mask = self.molMDP.block_nrs <= self.max_branches - len(molecule.stems) - 1
        if len(molecule.jbond_atmidxs) == self.max_blocks-1:
            jbonds_mask = np.zeros(self.num_blocks, dtype=np.bool)
        else:
            jbonds_mask = np.ones(self.num_blocks, dtype=np.bool)
        add_mask = np.logical_and(np.logical_and(atoms_mask, branches_mask), jbonds_mask)
        add_mask = np.asarray(add_mask, dtype=np.float32)
        add_mask = np.tile(add_mask[None, :], [self.max_branches, 1])
        num_stems = max(stem_fps_.shape[0], 1 if molecule.numblocks == 0 else 0)
        add_mask[num_stems:, :] = False
        add_mask = add_mask.reshape(-1)
        action_mask = np.concatenate([np.ones([1], dtype=np.float32), break_mask, add_mask])

        obs = {
            "mol_fp": mol_fp,
            "stem_fps": stem_fps,
            "jbond_fps": jbond_fps,
            "action_mask": action_mask,
            "num_steps": num_steps
               }
        return obs


class BlockMolEnv_v4:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems", "blockidxs"]

    def __init__(self, config):
        self.num_blocks = config["num_blocks"]
        self.max_blocks = config["max_blocks"]
        self.max_steps = config["max_steps"]
        self.max_simulations = config["max_simulations"]
        self.random_blocks = config["random_blocks"]
        #
        self.molMDP = MolMDP(**config["molMDP_config"])
        self.observ = FPObs_v1(config, self.molMDP)
        self.reward = config["reward"](**config["reward_config"])

        self.action_space = self.observ.action_space
        self.observation_space = self.observ.observation_space

    def _if_terminate(self):
        terminate = False
        # max steps
        if self.num_steps >= self.max_steps:
            terminate = True
        # max simulations
        if self.max_simulations is not None:
            if self.num_simulations >= self.max_simulations:
                terminate = True
        return terminate

    def reset(self):
        self.num_steps = 0
        self.num_simulations = 0
        self.molMDP.reset()
        self.molMDP.random_walk(self.random_blocks)
        self.reward.reset()
        return self.observ(self.molMDP.molecule, self.num_steps)

    def step(self, action):
        if (action == 0):
            simulate = True
            self.num_simulations += 1
        elif action <= (self.max_blocks - 1):
            simulate = False
            self.molMDP.remove_jbond(jbond_idx=action-1)
        else:
            simulate = False
            stem_idx = (action - self.max_blocks)//self.num_blocks
            block_idx = (action - self.max_blocks) % self.num_blocks
            self.molMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)

        self.num_steps += 1
        obs = self.observ(self.molMDP.molecule, self.num_steps)
        done = self._if_terminate()
        reward, info = self.reward(self.molMDP.molecule, simulate, done, self.num_steps)
        info["molecule"] = self.molMDP.molecule
        return obs, reward, done, info

    def get_state(self):
        mol_attr = {attr: deepcopy(getattr(self.molMDP.molecule, attr)) for attr in self.mol_attr}
        num_steps = deepcopy(self.num_steps)
        num_simulations = deepcopy(self.num_simulations)
        previous_reward = deepcopy(self.reward.previous_reward)
        mol = deepcopy(self.molMDP.molecule._mol)
        return mol_attr, num_steps, num_simulations, previous_reward, mol

    def set_state(self,state):
        mol_attr, self.num_steps, self.num_simulations, self.reward.previous_reward, self.molMDP.molecule._mol \
            = deepcopy(state)
        [setattr(self.molMDP.molecule, key, value) for key, value in mol_attr.items()]
        return self.observ(self.molMDP.molecule, self.num_steps)

    def render(self, outpath):
        mol = self.molMDP.molecule.mol
        if mol is not None: Chem.Draw.MolToFile(mol, outpath)


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


class BlockMolEnv_dummy:
    def __init__(self, config=None):
        self.mol_fp_len = config["mol_fp_len"]
        self.num_blocks = config["num_blocks"]
        self.stem_fp_len = config["stem_fp_len"]
        self.max_steps = config["max_steps"]
        self.max_branches = config["max_branches"]

        self.action_space = Discrete(self.max_branches * self.num_blocks + 1)
        self.observation_space = Dict({
            "mol_fp": Box(low=0, high=10, shape=(self.mol_fp_len,)),
            "stem_fps": Box(low=0, high=10, shape=(self.max_branches, self.stem_fp_len,)),
            "action_mask": Box(low=0, high=1, shape=(self.max_branches * self.num_blocks + 1,)),
            "step": Discrete(n=self.max_steps)
        })

        mol_fp = np.ones(shape=(self.mol_fp_len,),dtype=np.float32) * 2
        stem_fps = np.ones(shape=(self.max_branches, self.stem_fp_len), dtype=np.float32) * 3
        action_mask = np.zeros(shape=(self.max_branches * self.num_blocks + 1,))
        action_mask[:5] = 1
        self.obs = {"mol_fp": mol_fp, "stem_fps": stem_fps, "action_mask": action_mask, "step": 0}

        self.reset()

    def reset(self):
        self.reward = 0.0
        self.nsteps = 0
        return self.obs

    def step(self, action):
        if not (action % 4):
            done = True
            self.score = float(np.random.uniform(low=0.0, high=1,size=[]))
        else:
            done = False
            self.reward = 0.0
        info = {}
        self.nsteps += 1
        return self.obs, self.reward, done, info

    def set_state(self, state):
        mol_state, reward = state
        self.reward = reward
        self.nsteps = mol_state["nsteps"]
        return self.obs

    def get_state(self):
        molecule = {"nsteps": self.nsteps}
        return molecule, self.reward