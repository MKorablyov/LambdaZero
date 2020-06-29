import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import torch as th

def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def schwefel(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)

    result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))

    return result


def plot_traj():
    x = th.tensor((np.mgrid[-100:100,-100:100] / 100).reshape(2,-1).T)
    y = schwefel(x)
    x = x.numpy().T.reshape(2, 2*100, 2*100)
    y = y.numpy().reshape(2*100, 2*100)

    plt.subplot(2,2,1, projection="3d")
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.gca()
    ax.plot_surface(x[0], x[1], y, cmap='coolwarm', edgecolor='none',
                    rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    plt.savefig("/home/maksym/Desktop/traj.png",dpi=400)


    #plt.show()

#plot_traj()

n1 = np.random.normal(0, 1, size=1000)
n2 = np.random.normal(0, 1, size=1000)


print(np.matmul(n1,n1.T) / 1000)



#plt.scatter(n1,n2)
#plt.show()

#print(schwefel(x))

