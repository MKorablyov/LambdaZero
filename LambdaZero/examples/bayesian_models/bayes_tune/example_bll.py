class ModelWithUncertainty(nn.Module):
	""" This can be Bayesian Linear Regression, Gaussian Process, Ensemble etc...
	Also note before I have been thinking that these models have additional feature extractor, which acts on X
	first, this is still sensible to have here if we think necessary. eg this may be MPNN on SMILES
	"""
	def fit(self,x_train, y_train):
		pass

	def update(self,x_train, y_train):
		"""
		Some models allow you to update them more cheaply than retraining with all
		points (eg low rank update for Bayesian regression.) Hence you have an update
		function too.
		"""
		pass

	def get_predict_mean_and_variance_func(self):
		"""
		returns a function
		that should take in X_test [N, d] and return a two element tuple of tensor ([N,f], [N,f])
		which contains the predictive mean and variance at that output point
		f is the dimensionality of y, which is probably always going to be 1...?
		N is the batch size
		"""
		pass

	def get_predict_sample(self):
		"""
		This samples from your distribution over models/parameters/hyperparameters and returns
		an evaluator that tells you what the sampled model will predict at that point. Useful for
		eg Thompson sampling.
		returns a function
		that should take in X_test [N, d] and return a tensor [N,f]
		that is the predicted value at that point given the model parameters you have sampled
		f is the dimensionality of y, which is probably always going to be 1...?
		N is the batch size, d the dimensionality of features
		"""
		pass


class Acquirer:
    def __init__(self, available_points, model: BOModel, acquistion_function):
        """
        :param acquistion_function: The reason why this is a function is because you may want to
        change it as you gather points (ie start random and then switch to batch BALD etc...)
        """
        ...

    def update_available_points(self, seen_points):
        ...

    def get_batch_points_to_query(self, batch_size):
        ...


class Querier:
    """
    stores an oracle and the queried points from it.
    """

    def __init__(self, oracle_calling_function):
        ...

    @property
    def unseen_points(self) -> set:
        """
        Method only exists when you know all the points you can query.
        """
        ...

    @property
    def seen_points_and_properties(self):
        ...

    def query_properties(points_to_query: list) -> list:
        ...

# Then UCB looks like (very roughly...)
# eval_func = model.get_predict_mean_and_variance_func()
# all_scores = []
# for batch in test_points:
# 	# ^ this is the loop which you parallelize with eg Ray
# 	mean_on_batch, var_on_batch = eval_func(batch)
# 	score_on_batch = mean_on_batch + kappa * var_on_batch
# 	all_scores.append(score_on_batch)
# all_scores = concatenate(all_scores)
# next_point_to_query = np.argmax(all_scores)

# # And Thompson Sampling looks like (very roughly...)
# eval_func = model.get_predict_sample()
# all_scores = []
# for batch in test_points:
# 	# ^ this is the loop which you parallelize with eg Ray
# 	score_on_batch = eval_func(batch)
# 	all_scores.append(score_on_batch)
# all_scores = concatenate(all_scores)
# next_point_to_query = np.argmax(all_scores)

# # And Batch BALD looks something like (very very roughly...! eg a_batch_bold needs the xs  )
# eval_funcs = [model.get_predict_sample() for _ in range(k)]
# # ^ these funcs will be for the Monte Carlo estimator, eg see Section 3.3 of BatchBald paper
# acq_batch_so_far = []
# for i in range(acquisition_batch_size):
# 	all_scores = []
# 	for databatch in test_points:
# 		# ^ this is the loop which you parallelize with eg Ray
# 		score_on_databatch = a_batch_bold(databatch, acq_batch_so_far, eval_funcs)
# 		all_scores.append(score_on_databatch)
# 	all_scores = concatenate(all_scores)
# 	next_point_to_query = np.argmax(all_scores)
# 	acq_batch_so_far.append(next_point_to_query)