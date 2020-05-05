import os, sys, time, socket, os.path as osp
import numpy as np
from affinity_models.alpha_zero.environments.molecule import BlockMolEnv_v3
from affinity_models.rl_molecule.rlmol.env import MolMDP, QEDReward, PredDockReward

from copy import deepcopy
from matplotlib import pyplot as plt
from rdkit import Chem

from affinity_models.alpha_zero.examples.config import get_config

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "cfg000"
trainer, config, memory, summaries_dir, _ = get_config(config_name)
print("starting with config:", config_name)

env_config = config["env_config"]
env = config["env"](env_config)
obs = env.reset()

past_states = []
past_nact = []
past_smis = []
print(summaries_dir)
done = False
step = 0

while not done:
    actions = np.where(obs["action_mask"])[0]
    last = env.get_state()
    for action in actions[:10]:

        obs, reward, done, info = env.step(action)

        img_path = os.path.join(summaries_dir, str(step) + "_" + str(action) + ".png")
        env.render(img_path)
        #print("i")
        env.set_state(last)

    action = np.random.choice(actions[:10])
    print(action)
    obs, reward, done, info = env.step(action)

    print("step", step)
    step += 1



# start = time.time()
# for i in range(1000):
#     actions = np.where(obs["action_mask"])[0]
#     action = np.random.choice(actions)
#     obs, reward, done, info = env.step(action)
#
#     if done:
#         obs = env.reset()
#     else:
#         id = str(time.time())
#         s_ = env.get_state()
#         past_states.append((s_, (obs, reward, done, info), id))
#         #print("created id and state", id,s_)
#
#     if info["molecule"].mol is not None:
#         smi = Chem.MolToSmiles(info["molecule"].mol)
#     else:
#         smi = None
#     if not smi in past_smis: past_smis.append(smi)

    # if i % 100 == 99:
    #     print("step", i, "exps", "%.3f" % (100 / (time.time() - start)))
    #     start = time.time()
    #
    #     print(smi, len(past_smis))
    #     print("action", action,
    #           "done",  done,
    #           len(info["molecule"].slices),
    #           len(info["molecule"].stems,),
    #           "num steps", env.num_steps,
    #           "num avail actions", (obs["action_mask"].sum()))


    # if len(info["molecule"].slices) > 3:
    #     outpath = os.path.join("/home/maksym/model_summaries/MolBlock_v3", str(i) + ".png")
    #     env.render(outpath)
    # past_nact.append((obs["action_mask"].sum() - 1) / 105)

    # assert env.observation_space.contains(obs), "check observation"
    #
    # if len(past_states) > 20 and not done:
    #     state = past_states[np.random.choice(len(past_states))]
    #     print(env.molMDP.block_mols)
    #     print(state[0][0].keys())
    #     obs = env.set_state(state[0])
    #     print("set id and state", state[2], state[0])

# past_nact = np.asarray(past_nact)
# plt.hist(past_nact, bins=np.arange(past_nact.min(), past_nact.max()+1))
# plt.show()