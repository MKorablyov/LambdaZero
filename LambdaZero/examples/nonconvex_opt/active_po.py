import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from LambdaZero.environments import DummyNonConvex

from LambdaZero.environments.dummy_nonconvex import schwefel












# def plot_traj():
#     x = torch.tensor((np.mgrid[-100:100, -100:100] / 100).reshape(2, -1).T)
#     y = schwefel(x)
#     x = x.numpy().T.reshape(2, 2 * 100, 2 * 100)
#     y = y.numpy().reshape(2 * 100, 2 * 100)
#     plt.subplot(1, 1, 1, projection="3d")
#     ax = plt.gca()
#     ax.plot_surface(x[0], x[1], y, cmap='coolwarm', edgecolor='none',
#                     rstride=1, cstride=1, color='c', alpha=0.5, linewidth=0)
#     plt.savefig("/home/maksym/Desktop/traj.png", dpi=400)
# plot_traj()














# todo: priors in a very migh dimesional space should evolve slowly
# todo: some number of steps could be enough for the RL algorithm to achieve good performance
# todo: another idea is to be able to factor out updates in proxy: IE proxy is a task; one could sample many
# todo: at smallest best-found-so-far could become a factor

# todo:
# Active RL
# learn to walk in non-convex space
# start 0, 0
# walk -1,0,1 in each dimension
# observe embedding of coordinate

# reward at each step
# V1: usual reward
# V2: EI estimate from 1-layer model


# todo: I need to create an efficient representaiton which could fit the original function very well
# todo: diameter of the space ~15; each step ~1024 actions; ~10 dimensions; ~1 * 10^12 molecules total

