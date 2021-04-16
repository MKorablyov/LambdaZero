import numpy as np
import math

def satlins(x, cutoff0, cutoff1, eps_min=0.01):
    "shifted saturated linearity activation function _/-"
    x = (x - cutoff0) / (cutoff1 - cutoff0)
    x = min(max(eps_min, x), 1.0)  # relu to maxout at 1
    return x

def elu2(x, x_shift=-1., y_shift=1., epsilon=math.e):
    "this is elu which is always positive and 1 is 1 with default settings"
    x = np.asarray(x,dtype=np.float32) + x_shift
    act = []
    for val in x:
        if val > 0:
            act.append(float(y_shift + val))
        else:
            act.append(float(y_shift + (epsilon ** val-1)))
    return act




if __name__ == "__main__":
    from matplotlib import pyplot as plt
    x = np.random.uniform(low=-3,high=3,size=1000)
    #print(x)
    y = elu2(x)
    plt.scatter(x,y)
    plt.show()
    #print(elu2([-1]))