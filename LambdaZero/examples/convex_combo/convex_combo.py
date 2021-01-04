import time
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann, Levy, Michalewicz, Powell, Rastrigin
from gpytorch.mlls import ExactMarginalLogLikelihood


class cfg:
    num_init = 2500
    dim = 25
    mc_samples = 250
    #obj = Hartmann(dim=dim, negate=True)
    #obj = Levy(dim=dim)
    #obj = Michalewicz(dim=dim)
    #obj = Powell(dim=dim)
    obj = Rastrigin(dim=dim)
    # todo: with these man-made test functions a heuristic of sampling 10 points/dimension is good

# train on random data
train_x = torch.rand(cfg.num_init, cfg.dim)
train_obj = cfg.obj(train_x).unsqueeze(-1)
model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)


from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
best_value = train_obj.max()
ei = ExpectedImprovement(model=model, best_f=best_value)

from botorch.optim import optimize_acqf


new_point_analytic, v = optimize_acqf(
    acq_function=ei,
    bounds=torch.tensor([[0.0] * cfg.dim, [1.0] * cfg.dim]),
    q=1,
    num_restarts=200,
    raw_samples=500,
    options={},
)
print(new_point_analytic, v)

from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler


sampler = SobolQMCNormalSampler(num_samples=cfg.mc_samples, seed=0, resample=False)
MC_EI = qExpectedImprovement(model, best_f=best_value, sampler=sampler)

torch.manual_seed(seed=1) # to keep the restart conditions the same
new_point_mc, v = optimize_acqf(
    acq_function=MC_EI,
    bounds=torch.tensor([[0.0] * cfg.dim, [1.0] * cfg.dim]),
    q=1,
    num_restarts=200,
    raw_samples=500,
    options={},
    return_best_only=True
)
print(new_point_mc, v)
#print(v.shape)
#print(new_point_mc)


from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy





# todo: load dataset
# todo: train MPNN
# todo: save embeddings

###
# from botorch.gen import gen_candidates_scipy
# from botorch.optim.initializers import gen_batch_initial_conditions
# qEI = qExpectedImprovement(model=model, best_f=best_value)
# bounds = torch.tensor([[0.0] * cfg.dim, [1.0] * cfg.dim])
# Xinit = gen_batch_initial_conditions(qEI,
#                                      bounds=bounds,
#                                      q=10,
#                                      num_restarts=25,
#                                      raw_samples=500)
#
# batch_candidates, batch_acq_values = gen_candidates_scipy(
#     initial_conditions=Xinit,
#     acquisition_function=qEI,
#     lower_bounds=bounds[0],
#     upper_bounds=bounds[1],)
#
# print(Xinit.shape, batch_candidates.shape, batch_acq_values.shape)
#time.sleep(1000)
