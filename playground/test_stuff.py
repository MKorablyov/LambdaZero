import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("test_file", "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    txt = line.strip().split(" ")
    if len(txt) != 6 or not txt[2].isnumeric() or not txt[1].isnumeric():
        # print(line.strip())
        continue
    data.append([int(txt[1]), int(txt[2]), float(txt[3]), txt[4], txt[5]])

df = pd.DataFrame(data, columns=["seed", "action", "reward", "smiles", "next"])

(df.groupby("seed")["smiles"].nunique() == 1).all()
(df.groupby(["seed", "action"]).reward.nunique() == 1).all()
df.groupby("seed").size().describe()
df.iloc[:2048].groupby("seed").size().describe()
df.iloc[2*2048:3*2048].groupby("seed").size().describe()

ax = None
for i in range(3):
    ax = df.iloc[i*2048:(i+1)*2048].groupby("seed").size().plot(kind="bar", ax=ax)
