import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import torch as th
from gym.spaces import Discrete, Dict, Box



def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def schwefel(x, xmin=-1, xmax=1, yrange=2000):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    val = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    return val / yrange


class NonConvexOptEnv:
    def __init__(self, config):
        self.action_space = Dict({i:Discrete(4) for i in range(config["num_dims"])})
        self.observation_space = Dict({
            "onehot_embedding": Box(low=0, high=1, shape=(999,)),
        })

    def step(self):
        pass
        # return obs, reward, done, info

    def reset(self):
        pass



config = {
    "num_dims": 20,
    "discretize": 100,
}









# def plot_traj():
#     x = th.tensor((np.mgrid[-100:100,-100:100] / 100).reshape(2,-1).T)
#     y = schwefel(x)
#     x = x.numpy().T.reshape(2, 2*100, 2*100)
#     y = y.numpy().reshape(2*100, 2*100)
#     plt.subplot(2,2,1, projection="3d")
#     ax = plt.gca()
#     ax.plot_surface(x[0], x[1], y, cmap='coolwarm', edgecolor='none',
#                     rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
#
#     plt.savefig("/home/maksym/Desktop/traj.png",dpi=400)





