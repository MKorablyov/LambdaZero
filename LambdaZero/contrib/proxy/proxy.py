import time
import ray

class Proxy:
    def __init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_d, self.proposed_acq = proposed_x, proposed_d, proposed_acq
        self.logger = logger

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
        if len(self.proposed_x) == self.update_freq:
            self.acquire_and_update()
            self.proposed_x, self.proposed_acq, self.proposed_d = [], [], []
        return None

    def acquire_and_update(self):
        raise NotImplementedError

    def acquire(self, x):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError


class Actor():
    def __init__(self, scoreProxy, actor_sync_freq):
        self.scoreProxy = scoreProxy
        self.sync_freq = actor_sync_freq
        # initialize
        self.num_calls = 0
        self.acquisition_func = ray.get(scoreProxy.get_acquisition_func.remote())
        self.names = []

    def __call__(self, x, d):
        # compute acquisition value
        self.num_calls += 1
        acq, info = self.acquisition_func.acquisition_value(x)

        # send molecule to the remote proxy
        self.scoreProxy.propose_x.remote(x, acq, d)

        # sync weights with proxy if needed
        if self.num_calls % self.sync_freq==1:
            # todo - need to figure out to do non-blocking calls here
            self.acquisition_func = ray.get(self.scoreProxy.get_acquisition_func.remote())
        return acq, info