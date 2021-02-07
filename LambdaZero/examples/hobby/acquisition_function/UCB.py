import time
from .acquisition_function import AcquisitionFunction




class UCB(AcquisitionFunction):
    def __init__(self, config):
        AcquisitionFunction.__init__(self, config["model"], config["model_config"])

    def update_with_seen(self, x, y):
        # self.seen_x +=x
        # self.model_with_uncertainty.fit(x,y, self.val_x, self.val_y)
        pass

    def acquisition_value(self, x):
        # mean, var = self.model.get_mean_and_variance(molecules)
        # return mean + kappa * var
        return 1.0

    def acquire_batch(self, x, discounts, aq_values=None):
        # if aq_values = acquisition_values(x, self.model)
        # aq_values_ = aq_values[top_k]
        # return idx
        pass