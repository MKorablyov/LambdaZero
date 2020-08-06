import numpy as np
import time
import ray
from scipy.cluster.vq import kmeans2


@ray.remote
def get_fingerprint(mol):

    return np.random.normal(size=[100])



mols = np.arange(160)

ray.init()
fps = np.stack(ray.get([get_fingerprint.remote(m) for m in mols]),axis=0)


_, klabel = kmeans2(fps, k=20)

#print(klabel)

train_set, test_set = [],[]

for i in range(20):
    idx = np.where(klabel==i)[0]
    if np.random.uniform() < 0.1:
        test_set.append(idx)
    else:
        train_set.append(idx)

train_set = np.concatenate(train_set)
print(train_set)