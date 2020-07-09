



# class NormalizeScalar(th.Module):
    # init(values, mean=None, var=None)
    # normalize(batch)
    # unnormalize(batch)


# class MCdropout:
    # _(config)
    # self.model

#   fit(dataset):
        # do 3 epochs

#   predict_mean_variance(dataset)
        # do 10 epochs; compute mean and variance

#   def _train():
#









# class UCBAcquisition(tune.Trainable):
#   _setup(config):

    # config.bayesian_model
    # config.train_set


#   def _train():
    # idxs = self.acquire_batch()
    # self.update_with_seen()

#   update_with_seen([idxs])
    # if_acquired[idxs] = True
    # model.fit(data(Subset([if_acquired])))


#   acquire_batch()
    # model.predict_mean_variance(data[if_acquired])
    # scores = mean + (kappa * variance)
    # return idxs