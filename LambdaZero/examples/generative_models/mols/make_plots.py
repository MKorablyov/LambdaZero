import numpy as np

from LambdaZero.utils import get_external_dirs


runs = ["proxy__7"]

import gzip,pickle




with gzip.open('results/proxy__7/info.pkl.gz', 'rb') as f: data = pickle.load(f)

# print(len(data))
# print(type(data))
# print(data.keys())
# print(data["batch_metrics"])
# print("num batches", len(data["batch_metrics"]))
#
#
# #print(data["batch_metrics"][-1]["reward_max"])
# #print(np.mean([d < 0.7 for d in data["batch_metrics"][-1]["dists"]]))
#
# for d in data["batch_metrics"]:
#     print(d["reward_mean"])
