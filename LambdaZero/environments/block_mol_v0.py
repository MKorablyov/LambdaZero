import numpy as np
from gym.spaces import Discrete, Dict, Box

class BlockMolEnv_v0:
    # dummy molecule environment
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