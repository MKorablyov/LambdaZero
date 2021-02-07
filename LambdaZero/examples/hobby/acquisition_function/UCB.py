def acquisition_values_UCB(model, x):
    # mean, var = self.model.get_mean_and_variance(molecules)
    # return mean + kappa * var
    pass

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

    def acquire_batch(self, x, discounts, aq_values=None):
        # if aq_values = _acquisition_values(x, self.model)
        # aq_values_ = aq_values[top_k]
        # return idx
        pass