import time
import ray

class Proxy:
    def __init__(self, update_freq, proposed_x,  proposed_y, proposed_d):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_y, self.proposed_d = proposed_x, proposed_y, proposed_d

    def propose_x(self,x, y, d=1):
        """
        :param x: point to evaluate
        :param y: current estimate of the y value
        :param d: discount factor defaults to 1. This would be used for the acquisition
        :return:
        """
        self.proposed_x.append(x)
        self.proposed_y.append(y)
        self.proposed_d.append(d)
        if len(self.proposed_x) == self.update_freq:
            self.acquire_and_update() #
            self.proposed_x, proposed_y, proposed_d = [], [], []
            # todo: report statistics
        return None

    def acquire_and_update(self):
        raise NotImplementedError

    def acquire(self, x):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError



class Actor():
    def __init__(self, scoreProxy, update_freq):
        self.scoreProxy = scoreProxy
        self.update_freq = update_freq
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
        aq_val = self.acqusition_func.acquisition_value(x)

        # send molecule to the remote proxy
        self.scoreProxy.propose_x.remote(x, aq_val, d)

        # sync weights with proxy if needed
        if self.num_calls % self.update_freq ==0:
            # fixme - need to figure out to do non-blocking calls
            self.acqusition_func = ray.get(self.scoreProxy.get_acquisition_func.remote())

        # graph = self.proc_func(x) # this will build graph
        # aq_val = self.aq_values_func(graph)
        # self.sharex_func(x)
        # if self.sync_weigths

        # return aq_val

        #y = 1.0 # fixme -- let's call y_, y_ac and y ground truth and estimates
        print("called actor:", x, aq_val)
        return aq_val