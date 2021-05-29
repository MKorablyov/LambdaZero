import matplotlib.pyplot as plt
import numpy as np

from LambdaZero.utils import get_external_dirs
import gzip, pickle

import matplotlib

runs = ["proxy__0", "proxy__1", "proxy__2", "proxy__3", "proxy__4", "proxy__5", "proxy__6",
        "proxy__7", "proxy__8", "proxy__9", "proxy__10", "proxy__11"]

legend =  ["first_acquisition 1024, reward_exp 10",
           "first_acquisition 512, reward_exp 10",
           "first_acquisition 2048, reward_exp 10",
           "first_acquisition 4096, reward_exp 10",

           "first_acquisition 1024, reward_exp 4",
           "first_acquisition 512, reward_exp 4",
           "first_acquisition 2048, reward_exp 4",
           "first_acquisition 4096, reward_exp 4",

           "first_acquisition 1024, reward_exp 2",
           "first_acquisition 512, reward_exp 2",
           "first_acquisition 2048, reward_exp 2",
           "first_acquisition 4096, reward_exp 2",
           ]



run_idxs = [1,0,2,3,
            5,4,6,7,
            9,8,10,11]
runs = [runs[idx] for idx in run_idxs]
legend = [legend[idx] for idx in run_idxs]

# # reward_mean
# for run in runs:
#     run_path = 'results/{}/info.pkl.gz'.format(run)
#     with gzip.open(run_path, 'rb') as f:
#         data = pickle.load(f)
#     reward_mean = [bm["reward_mean"] for bm in data["batch_metrics"]]
#     plt.plot(reward_mean)
# plt.xlabel("acquisition_step")
# plt.ylabel("reward_mean")
# plt.legend(legend)
# plt.savefig("/home/maksym/Desktop/reward_mean.png", dpi=300)
# plt.close()
#
# # reward_max
# for run in runs:
#     run_path = 'results/{}/info.pkl.gz'.format(run)
#     with gzip.open(run_path, 'rb') as f:
#         data = pickle.load(f)
#     reward_mean = [bm["reward_max"] for bm in data["batch_metrics"]]
#     plt.plot(reward_mean)
# plt.xlabel("acquisition_step")
# plt.ylabel("reward_max")
# plt.legend(legend)
# plt.savefig("/home/maksym/Desktop/reward_max.png", dpi=300)
# plt.close()
#
# # tanimoto_mean
# for run in runs:
#     run_path = 'results/{}/info.pkl.gz'.format(run)
#     with gzip.open(run_path, 'rb') as f:
#         data = pickle.load(f)
#     reward_mean = [bm["dists_mean"] for bm in data["batch_metrics"]]
#     plt.plot(reward_mean)
# plt.xlabel("acquisition_step")
# plt.ylabel("Tanimoto dist mean")
# plt.legend(legend)
# plt.savefig("/home/maksym/Desktop/dists_mean.png", dpi=300)
# plt.close()
#
# # diversity
# for run in runs:
#     run_path = 'results/{}/info.pkl.gz'.format(run)
#     with gzip.open(run_path, 'rb') as f:
#         data = pickle.load(f)
#     some = [np.mean(np.array(bm["dists"]) < 0.6) for bm in data["batch_metrics"]]
#     plt.plot(some)
# plt.xlabel("acquisition_step")
# plt.ylabel("diversity mean")
# plt.legend(legend)
# plt.savefig("/home/maksym/Desktop/diversity_mean.png", dpi=300)
# plt.close()

import rdkit.DataStructs
from rdkit import Chem



def get_tanimoto_pairwise_mean(mols):
    fps = [Chem.RDKFingerprint(i.mol) for i in mols]
    pairwise_sim = []
    for i in range(len(mols)):
        pairwise_sim.extend(rdkit.DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]+fps[i+1:]))
    return pairwise_sim



class NumModes:
    def __init__(self, reward_exp, reward_norm, reward_thr=8, tanimoto_thr=0.7):
        self.reward_exp = reward_exp
        self.reward_norm = reward_norm
        self.reward_thr = reward_thr
        self.tanimoto_thr = tanimoto_thr

        self.modes = []
        self.max_reward = -1000

    def __call__(self, batch):
        candidates = []
        for some in batch:
            reward, mol = some[0], some[1]
            reward = (reward ** (1/self.reward_exp)) * self.reward_norm
            if reward > self.max_reward: self.max_reward = reward
            if reward > self.reward_thr:
                candidates.append(mol)

        if len(candidates) > 0:
            # add one mode if needed
            if len(self.modes)==0: self.modes.append(Chem.RDKFingerprint(candidates[0].mol))

            for mol in candidates:
                fp = Chem.RDKFingerprint(mol.mol)
                sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))

                if all(sims < self.tanimoto_thr):
                    self.modes.append(fp)

        return self.max_reward, len(self.modes)



def compute_diversity(run_path, reward_exp, reward_norm, reward_thr, skip=100):
    with gzip.open(run_path, 'rb') as f:
        data = pickle.load(f)
    numModes = NumModes(reward_exp, reward_norm, reward_thr=reward_thr)
    diversity = []
    mean_tanimoto_sim = []
    num_modes = []
    for i in range(len(data)//256):  # 4000
        if not i % skip:
            batch = data[i * 256: ((i + 1) * 256)]
            # mols = [x[1] for x in batch]
            # tm = get_tanimoto_pairwise_mean(mols)
            # diversity.append(np.mean(np.array(tm) < 0.75))
            # mean_tanimoto_sim.append(np.mean(tm))

            max_reward, nm = numModes(batch)
            num_modes.append(nm)
            print("i", i, max_reward, nm)

    x = np.linspace(start=0, stop=1000000, num=len(num_modes))
    return x, diversity, mean_tanimoto_sim, num_modes



# ours_x, ours_diversity, ours_sim, ours_modes = compute_diversity('/home/maksym/Datasets/temp/ours_sampled_mols.pkl.gz', 10, 8, 7.)
# mars_x, mars_diversity, mars_sim, mars_modes = compute_diversity('/home/maksym/Datasets/temp/mars_sampled_mols.pkl.gz', 8, 1, 7.)
# ppo_x, ppo_diversity, ppo_sim, ppo_modes = compute_diversity('/home/maksym/Datasets/temp/ppo_sampled_mols.pkl.gz', 4, 10, 7.)
# plt.plot(ours_x, ours_modes, color=matplotlib.cm.tab10(0))
# plt.plot(mars_x, mars_modes, color=matplotlib.cm.tab10(1))
# plt.plot(ppo_x, ppo_modes, color=matplotlib.cm.tab10(2))

ours_x, ours_diversity, ours_sim, ours_modes = compute_diversity('/home/maksym/Datasets/temp/ours_sampled_mols.pkl.gz', 10, 8, 8.)
mars_x, mars_diversity, mars_sim, mars_modes = compute_diversity('/home/maksym/Datasets/temp/mars_sampled_mols.pkl.gz', 8, 1, 8.)
ppo_x, ppo_diversity, ppo_sim, ppo_modes = compute_diversity('/home/maksym/Datasets/temp/ppo_sampled_mols.pkl.gz', 4, 10, 8.)
plt.plot(ours_x, ours_modes, color=matplotlib.cm.tab10(0), linestyle="--")
plt.plot(mars_x, mars_modes, color=matplotlib.cm.tab10(1), linestyle="--")
plt.plot(ppo_x, ppo_modes, color=matplotlib.cm.tab10(2), linestyle="--")


#plt.plot(ppo_modes, color=matplotlib.cm.tab10(2), linestyle="--")


plt.legend(["ours", "mars", "ppo"]) #, "ours, modes>8", "mars, modes>8", "ppo, modes>8"])

plt.xscale("log")
plt.xlabel("states visited")
plt.ylabel("num modes > 8")
plt.grid()

plt.savefig("/home/maksym/Desktop/num_modes_8.png",dpi=300)

