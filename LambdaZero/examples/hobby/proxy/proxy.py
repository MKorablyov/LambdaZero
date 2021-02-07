class Proxy:
    def __init__(self, update_freq, proposed_x,  proposed_y, proposed_d):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_y, self.proposed_d = proposed_x, proposed_y, proposed_d

    def propose_x(self,x, y, d):
        # todo having a discount is specialized
        self.proposed_x.append(x)
        self.proposed_y.append(y)
        self.proposed_d.append(d)

        if len(self.proposed_x) == self.update_freq:
            self.acquire_and_update() # todo: report statistics
            self.proposed_x, proposed_y = [], []
        return None

    def acquire_and_update(self):
        raise NotImplementedError

    def acquire(self, x):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def get_actor(self):
        raise NotImplementedError


class Actor():
    def __init__(self, proc_func, aq_values_func, get_weights_func, propose_x_func):
        self.proc_func = proc_func
        self.aq_values_func = aq_values_func
        self.get_weights = get_weights_func
        self.share_x_func = propose_x_func
        self.weights = get_weights_func()
        self.num_calls = 0

    def __call__(self, x):
        # graph = self.proc_func(x) # this will build graph
        # aq_val = self.aq_values_func(graph)
        # self.sharex_func(x)
        # if self.sync_weigths

        # return aq_val
        pass