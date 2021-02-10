import sys, time
from .model_with_uncertainty import ModelWithUncertainty





class MolFP(ModelWithUncertainty):
    def __init__(self):
        ModelWithUncertainty.__init__(self)
    num_fit = 0
    start = time.time()

    def fit(self,x,y):
        print("fit not implemented")
        print("mol graph", x[0])

        # 0.48 for no fit
        self.num_fit +=1
        print("exps:", "%.3f" % (self.num_fit / (time.time() - self.start)))

        #raise NotImplementedError

    def get_mean_and_variance(self,x):
        # todo:
        #print("mean variance not implemented", x)

        mean = [0.1 for i in x]
        var = [0.2 for i in x]
        return mean, var