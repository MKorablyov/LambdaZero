import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _exp_var(y_true,y_hat):
    var0 = ((y_true - y_true.mean()) ** 2).mean()
    exp_var = (var0 - ((y_true - y_hat) ** 2).mean()) / var0
    return exp_var

data = pd.read_csv("/home/maksym/Datasets/postera/activity_data.csv")


f_actives = np.array([d < 10 for d in data["f_avg_IC50"]])
r_actives = np.array([d < 10 for d in data["r_avg_IC50"]])
if_actives = np.logical_and(f_actives,r_actives)
actives = data[if_actives]

r_decoys = np.array([d < 1 for d in data["r_inhibition_at_50_uM"]])
f_decoys = np.array([d < 1 for d in data["r_inhibition_at_50_uM"]])
if_decoys = np.logical_and(f_decoys,r_decoys)
decoys = data[if_decoys]

clean_data = pd.concat([actives,decoys])
#plt.scatter(actives["f_avg_IC50"], actives["r_avg_IC50"])
#plt.scatter(decoys["f_avg_IC50"], decoys["r_avg_IC50"])
plt.scatter(np.argsort(actives["f_avg_IC50"].to_numpy()), np.argsort(actives["r_avg_IC50"].to_numpy()))
plt.show()

f_ic50, r_ic50 = actives["f_avg_IC50"].to_numpy(), actives["r_avg_IC50"].to_numpy()
print(_exp_var(r_ic50, r_ic50))


