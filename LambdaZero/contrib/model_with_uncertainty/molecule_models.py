from .model_with_uncertainty import ModelWithUncertainty


class MolFP(ModelWithUncertainty):
    def __init__(self):
        ModelWithUncertainty.__init__(self)

    def fit(self,x,y):
        print("fit not implemented")
        #raise NotImplementedError

    def get_mean_and_variance(self,x):
        # todo:
        mean = [0.1 for i in x]
        var = [0.2 for i in x]
        return mean, var