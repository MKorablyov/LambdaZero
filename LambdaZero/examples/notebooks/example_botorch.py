import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

d = 5

bounds = torch.stack([-torch.ones(d), torch.ones(d)])
train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(50, d)
train_Y = 1 - torch.norm(train_X, dim=-1, keepdim=True)
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)



# import math
# import torch
# from botorch.fit import fit_gpytorch_model
# from botorch.models import SingleTaskGP
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.acquisition import UpperConfidenceBound
# import cma
# import numpy as np


# X = torch.rand(20, 2) - 0.5
# Y = (torch.sin(2 * math.pi * X[:, 0]) + torch.cos(2 * math.pi * X[:, 1])).unsqueeze(-1)
# Y += 0.1 * torch.randn_like(Y)
#
# gp = SingleTaskGP(X, Y)
# mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
# fit_gpytorch_model(mll)
#
#
# UCB = UpperConfidenceBound(gp, beta=0.1)
#
#
# # get initial condition for CMAES in numpy form
# # note that CMAES expects a different shape (no explicit q-batch dimension)
# x0 = np.random.rand(2)
#
# # create the CMA-ES optimizer
# es = cma.CMAEvolutionStrategy(
#     x0=x0,
#     sigma0=0.2,
#     inopts={'bounds': [0, 1], "popsize": 50},
# )
#
# # speed up things by telling pytorch not to generate a compute graph in the background
# with torch.no_grad():
#
#     # Run the optimization loop using the ask/tell interface -- this uses
#     # PyCMA's default settings, see the PyCMA documentation for how to modify these
#     while not es.stop():
#         xs = es.ask()  # as for new points to evaluate
#         # convert to Tensor for evaluating the acquisition function
#         X = torch.tensor(xs, device=X.device, dtype=X.dtype)
#         # evaluate the acquisition function (optimizer assumes we're minimizing)
#         Y = - UCB(X.unsqueeze(-2))  # acquisition functions require an explicit q-batch dimension
#         y = Y.view(-1).double().numpy()  # convert result to numpy array
#         #print(xs)
#         es.tell(xs, y)  # return the result to the optimizer
#
# # convert result back to a torch tensor
# best_x = torch.from_numpy(es.best.x).to(X)
# print(es.best.f)
# print(best_x)














# import torch
# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_model
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.acquisition import UpperConfidenceBound
# from botorch.optim import optimize_acqf
#
# train_X = torch.rand(10, 2)
# Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
# Y += 0.1 * torch.rand_like(Y)
# train_Y = (Y - Y.mean()) / Y.std()
#
# print(train_X, train_Y)
# gp = SingleTaskGP(train_X, train_Y)
# mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
# fit_gpytorch_model(mll)
#
# UCB = UpperConfidenceBound(gp, beta=0.1)
#
# bounds = torch.stack([torch.zeros(2), torch.ones(2)])
# candidate, acq_value = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
#
# print(candidate, acq_value)