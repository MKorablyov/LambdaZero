import math
import torch

from botorch.test_functions import Branin
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(7)

bounds = torch.tensor(Branin._bounds).T
train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(10, 2)
train_Y = Branin(negate=True)(train_X).unsqueeze(-1)

train_X = normalize(train_X, bounds=bounds)
train_Y = standardize(train_Y + 0.05 * torch.randn_like(train_Y))

model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)


from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
candidate_set = torch.rand(1000, bounds.size(1), device=bounds.device, dtype=bounds.dtype)
candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
qMES = qMaxValueEntropy(model, candidate_set, use_gumbel=False, num_mv_samples=11)


from botorch.optim import optimize_acqf
# for q = 1
candidates, acq_value = optimize_acqf(
    acq_function=qMES,
    bounds=bounds,
    q=1,
    num_restarts=1,
    raw_samples=512,
)
print(candidates, acq_value)


#
#
# # for q = 2, sequential optimization
# candidates_q2, acq_value_q2 = optimize_acqf(
#     acq_function=qMES,
#     bounds=bounds,
#     q=2,
#     num_restarts=10,
#     raw_samples=512,
#     sequential=True,
# )
# print(candidates_q2, acq_value_q2)




# from botorch.optim import optimize_acqf_cyclic
#
# # for q = 2, cyclic optimization
# candidates_q2_cyclic, acq_value_q2_cyclic = optimize_acqf_cyclic(
#     acq_function=qMES,
#     bounds=bounds,
#     q=2,
#     num_restarts=10,
#     raw_samples=512,
#     cyclic_options={"maxiter": 2}
# )
# print(candidates_q2_cyclic, acq_value_q2_cyclic)