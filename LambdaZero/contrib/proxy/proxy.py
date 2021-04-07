import time
import ray
from copy import deepcopy

class Proxy:
    def __init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_d, self.proposed_acq = proposed_x, proposed_d, proposed_acq
        self.logger = logger
        self._model_id = 0

    def propose_x(self,x, d, acq):
        """
        :param x: points to evaluate
        :param aq: current estimates of the acquisition value at x
        :param d: discount factors defaults to 1. This would be used for the acquisition
        :return:
        """
        self.proposed_x.extend(x)
        self.proposed_d.extend(d)
        self.proposed_acq.extend(acq)
        # update model and acquisition function if needed
        if len(self.proposed_x) >= self.update_freq:
            # todo: a much better solution would be to re-compute acqusition values of data that has
            # arrived while this was retraining
            #proposed_x, proposed_d, proposed_acq = self.proposed_x, self.proposed_d, self.proposed_acq
            #self.proposed_x, self.proposed_d, self.proposed_acq = [], [], []
            self.acquire_and_update(self.proposed_x, self.proposed_d, self.proposed_acq)
            self.proposed_x, self.proposed_d, self.proposed_acq = [], [], []
            self._model_id += 1
        return None

    def acquire_and_update(self, proposed_x, proposed_d, proposed_acq):
        raise NotImplementedError

    def acquire(self, x):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_model_id(self):
        return self._model_id


class Actor():
    def __init__(self, scoreProxy, actor_sync_freq):
        self.scoreProxy = scoreProxy
        self.sync_freq = actor_sync_freq
        # initialize
        self.num_calls = 0
        self.acquisition_func = ray.get(scoreProxy.get_acquisition_func.remote())
        self.names = []
        self._acq_update_id = 0
        self._model_id = ray.get(self.scoreProxy.get_model_id.remote())

    def __call__(self, x, d):
        # compute acquisition value
        self.num_calls += 1
        acq, info = self.acquisition_func.acquisition_value(x)

        # send molecule to the remote proxy
        self.scoreProxy.propose_x.remote(x, d, acq)

        # sync weights with proxy if needed
        if self.num_calls % self.sync_freq==1:
            # todo - need to figure out to do non-blocking calls here
            self.acquisition_func = ray.get(self.scoreProxy.get_acquisition_func.remote())
            self._model_id = ray.get(self.scoreProxy.get_model_id.remote())

            self._acq_update_id += 1

        info["acq_update_id"] = self._acq_update_id
        info["model_id"] = self._model_id

        return acq, info