import numpy as np
import pandas as pd
# make dataset

sigma = []
x = []
for i in range(10000):
    sample_sigma = np.random.uniform(low=0.0, high=3.0)
    sample_x = np.random.normal(scale=sample_sigma)

    sigma.append(sample_sigma)
    x.append(sample_x)

sigma = np.asarray(sigma)
x = np.asarray(x)

# PDF = (1 / sigma * sqrt(2 pi))  *  exp -0.5 ((x-mu)/sigma)**2
# log PDF =  0.5 * sqrt(2 pi) * sigma  * ((x-mu)/sigma)**2

logP1 = 0.5 * (2 * np.pi) ** 0.5 * (sigma * (x / sigma) **2).mean()
np.random.shuffle(sigma)
logP2 = 0.5 * (2 * np.pi) ** 0.5 * (sigma * (x / sigma) **2).mean()

print("good / bad variance estimator log P", logP1, "/", logP2)