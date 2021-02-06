import numpy as np
from ray import tune


# class EnsembleMLP:
    # make ensemble of weights in torch
    #

    # def fit(x, y, x_val, y_val)
        # this would fit to the dataset

    # def get_mean_and_variance()
        # samples = ensemble(w) @ X
        # return  mean, variance(samples)

class UCB:
    # todo: acquisition function - maybe create classes
    #  AcqusitionFunction; ModelWithUncertainty
    def __init__(self):
        # make model
        # seen_x, seen_y, val_x, val_y = None
        pass

    def update_with_seen(self, x, y):
        # self.seen_x +=x
        # self.model_with_uncertainty.fit(x,y, self.val_x, self.val_y)
        pass

    def acqusition_values(self, x):
        # mean, var = self.model.get_mean_and_variance(molecules)
        # return mean + kappa * var
        pass

    def acquire_batch(self, x, discounts, aq_values=None):
        # if aq_values = self.compute_acquisition_values(x)
        # aq_values_ = aq_values[top_k]
        # return idx
        pass


class UCBTrainable(UCB, tune.Trainable):
    # load dataset of fingerprints

    # self.dataset = load_dataset()
    # _train()

        # return {"acc":0.1}
    pass



def load_dataset():
    x = np.random.uniform(size=(10000,1024))
    func = np.random.uniform(size=(1024,1)) / 1024.
    y = np.matmul(x, func)
    return x,y

load_dataset()


# tune run reference here:
# https://github.com/MKorablyov/LambdaZero/blob/master/LambdaZero/examples/bayesian_models/bayes_tune/UCB.py