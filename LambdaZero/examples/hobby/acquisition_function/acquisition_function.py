class AcquisitionFunction:
    def __init__(self, model, model_config):
        self.seen_x, self.seen_y, self.val_x, self.val_y = [],[],[],[]
        self.model = model(**model_config)

    def update_with_seen(self,x,y):
        raise NotImplementedError

    def acquire_batch(self,x, discounts, acq=None):
        raise NotImplementedError