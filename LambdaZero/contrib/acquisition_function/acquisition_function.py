import time
from typing import NewType


class AcquisitionFunction:
    def __init__(self, model, model_config:dict, acq_size:int):
        """
        Generic class for the acquisition functions
        :param model: model with uncertainty should have fit() and get_samples() at smallest
        :param model_config: config dictionary how to initialize model with uncertainty
        :param acq_size: int - how many datapoint will be acquired
        """
        self.acq_size = acq_size

    def update_with_seen(self,x,y):
        raise NotImplementedError

    def acquisition_value(self,x):
        raise NotImplementedError

    def acquire_batch(self, x, d, acq=None):
        raise NotImplementedError