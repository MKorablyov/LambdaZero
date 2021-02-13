class ModelWithUncertainty:
    def __init__(self):
        # todo: maybe it could be initialized with x_val, y_val
        pass

    def get_samples(self,x, num_samples):
        raise NotImplementedError

    def get_mean_and_variance(self,x):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError