


# make epochs
self.train_epoch = config["train_epoch"]
self.eval_epoch = config["eval_epoch"]


def _train(self):
    train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
    eval_scores = self.eval_epoch(self.val_set, self.model, self.device, self.config)
    # rename to make scope
    train_scores = [("train_" + k, v) for k, v in train_scores.items()]
    eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
    scores = dict(train_scores + eval_scores)
    return scores