import os, sys, time
path = \
    "/home/maksym/Datasets/generated/mols_new_target/ppo_bayes_reward_0082020-12-05_12-43-26"

print(len((os.listdir(path))))

import numpy as np
import gzip




#arr = np.load(os.path.join(path,"batch-155.pkl"),allow_pickle=True)



#print(max([a[1] for a in arr]))