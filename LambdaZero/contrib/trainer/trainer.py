from ray import tune

class BasicTrainer(tune.Trainable):
    def setup(self, config):
        self.train_x, self.train_y, self.val_x, self.val_y = config["load_data"](**config["load_data_config"])
        self.model = config["model"](**config["model_config"])
        self.logger = config["model_config"]["logger"]

    def step(self):
        self.model.fit(self.train_x, self.train_y)
        metrics = self.model.eval(self.val_x, self.val_y)
        return metrics