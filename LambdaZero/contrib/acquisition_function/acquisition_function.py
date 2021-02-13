import time
class AcquisitionFunction:
    def __init__(self, model, model_config, acq_size):
        self.acq_size = acq_size

    def update_with_seen(self,x,y):
        raise NotImplementedError

    def acquisition_value(self,x):
        raise NotImplementedError

    def acquire_batch(self, x, d, acq=None):
        raise NotImplementedError