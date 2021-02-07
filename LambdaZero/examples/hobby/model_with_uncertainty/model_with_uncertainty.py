class ModelWithUncertainty:
    def __init__(self):
        # todo: maybe it could be initialized with x_val, y_val
        pass

    def get_samples(self,x):
        raise NotImplementedError

    def get_mean_and_variacne(self,x):
        raise NotImplementedError

    def fit(self,x_train,y_train, x_val, y_val):
        raise NotImplementedError