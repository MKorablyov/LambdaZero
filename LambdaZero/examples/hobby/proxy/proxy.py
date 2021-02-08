import time
import ray

class Proxy:
    def __init__(self, update_freq, proposed_x,  proposed_acq, proposed_d):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_acq, self.proposed_d = proposed_x, proposed_acq, proposed_d

    def propose_x(self,x, acq, d):
        """
        :param x: point to evaluate
        :param aq: current estimate of the acquisition value at x
        :param d: discount factor defaults to 1. This would be used for the acquisition
        :return:
        """
        self.proposed_x.append(x)
        self.proposed_acq.append(acq)
        self.proposed_d.append(d)
        if len(self.proposed_x) == self.update_freq:
            self.acquire_and_update() #
            self.proposed_x, self.proposed_acq, self.proposed_d = [], [], []
            # todo: report statistics
        return None

    def acquire_and_update(self):
        raise NotImplementedError

    def acquire(self, x):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError


class Actor():
    def __init__(self, scoreProxy, sync_freq):
        self.scoreProxy = scoreProxy
        self.sync_freq = sync_freq
        # initialize
        self.num_calls = 0
        self.acqusition_func = ray.get(scoreProxy.get_acquisition_func.remote())

        #self.proc_func = proc_func
        #self.aq_values_func = aq_values_func
        #self.get_weights = get_weights_func
        #self.propose_x_func = propose_x_func
        #self.weights = get_weights_func()

    def __call__(self, x, d):
        # compute acquisition value
        self.num_calls += 1
        acq = self.acqusition_func.acquisition_value(x)

        # send molecule to the remote proxy
        self.scoreProxy.propose_x.remote(x, acq, d)

        # sync weights with proxy if needed
        if self.num_calls % self.sync_freq==0:
            # todo - need to figure out to do non-blocking calls here
            self.acqusition_func = ray.get(self.scoreProxy.get_acquisition_func.remote())

        # graph = self.proc_func(x) # this will build graph
        # aq_val = self.aq_values_func(graph)
        # self.sharex_func(x)
        # if self.sync_weigths
        return acq