import pandas as pd
import numpy as np
import torch as th
from matplotlib import pyplot as plt
import seaborn as sns

class cfg:
    # data could be found here: https://github.com/MKorablyov/brutal_dock/tree/master/d4/raw
    data = "/home/maksym/Datasets/brutal_dock/d4/raw/d4_100k_clust.parquet"


df = pd.read_parquet(cfg.data)
print(df.columns)

#sns.distplot(df["dockscore"])
#plt.show()



# total_budget = 3000

# strategies
# S1: greedy
# batch_size = 500, num_interations = 6

# class Model:
#   def train(states):
#   return MSE
#   def evaluate(states, labels):
#   return predictions, MSE

