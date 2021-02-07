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