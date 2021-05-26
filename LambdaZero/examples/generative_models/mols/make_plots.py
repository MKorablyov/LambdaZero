import numpy as np

from LambdaZero.utils import get_external_dirs


# exp_dir = f'{args.save_path}/proxy_{args.array}_{args.run}/'
# {exp_dir}/info.pkl.gz

#info = np.loadz("results/proxy__0/info.pkl.gz")

import gzip,pickle

with gzip.open('results/proxy__1/info.pkl.gz', 'rb') as f:
    data = pickle.load(f)

print(len(data))
print(type(data))
print(data.keys())
print(data["batch_metrics"])
print("num batches", len(data["batch_metrics"]))

#print(data["batch_metrics"][-1]["reward_max"])
#print(np.mean([d < 0.7 for d in data["batch_metrics"][-1]["dists"]]))


#print(1)
#print(info)