import time
import numpy as np
import torch
from LambdaZero.contrib.acquisition_function import AcquisitionFunction
from torch.distributions import Normal

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.cholesky import psd_safe_cholesky
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.models.utils import check_no_nans

CLAMP_LB = 1.0e-8

class MES(AcquisitionFunction):
    def __init__(self, model, model_config, acq_size):
        self.model = model(**model_config)
        num_y_samples = 20
        self.sampler = SobolQMCNormalSampler(num_y_samples)
        self.num_mv_samples = 10
        self.num_fantasies = 1
        self.gp_fitted = False
        AcquisitionFunction.__init__(self, model, model_config, acq_size)

    def update_with_seen(self, x, y):
        self.model.fit(x,y)
        print('######## gp fitted: ', self.gp_fitted)
        if self.gp_fitted is False:
            self.candidate_set = self.model.get_embed(x)[-10:, :]
            self.gp = SingleTaskGP(self.candidate_set, torch.tensor(y).unsqueeze(-1)[-10:, :])
            self.gp_fitted = True
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(self.mll)
        # todo: evaluate acquisition function on a separate train/val set
        return self.gp

    def acquisition_value(self, x):
        posterior = self.gp.posterior(self.model.get_embed(x).unsqueeze(1).unsqueeze(-3), observation_noise=False)
        mean = posterior.mean.squeeze(-1).squeeze(-1)
        variance = posterior.variance.clamp_min(CLAMP_LB).view_as(mean)
        check_no_nans(mean)
        check_no_nans(variance)

        if variance.dim() == 2:
            covar_mM = variance.unsqueeze(-1)
        else:
            covar_mM = variance

        self._sample_max_values()

        ig = self._compute_information_gain(
            X=self.model.get_embed(x).unsqueeze(1), mean_M=mean, variance_M=variance, covar_mM=covar_mM
        )

        return ig.mean(dim=0).detach().numpy()


    def acquire_batch(self, x, d, acq=None):

        if acq is not None:
            acq = self.acquisition_value(x)

        # compute indices with highest acquisition values
        idx = np.argsort(acq)[-self.acq_size:]
        # take selected indices
        x_ = [x[i] for i in idx]
        d_ = [d[i] for i in idx]
        acq_ = [acq[i] for i in idx]
        return x_, d_, acq_

    def _compute_information_gain(self, X, mean_M, variance_M, covar_mM):

        # Needs to add observation noise to the posterior
        posterior_m = self.gp.posterior(X.unsqueeze(-3), observation_noise=True)
        mean_m = posterior_m.mean.squeeze(-1)
        variance_m = posterior_m.mvn.covariance_matrix
        check_no_nans(variance_m)

        samples_m = self.sampler(posterior_m).squeeze(-1)
        L = psd_safe_cholesky(variance_m)
        temp_term = torch.cholesky_solve(covar_mM.unsqueeze(-1), L).transpose(-2, -1)

        mean_pt1 = torch.matmul(temp_term, (samples_m - mean_m).unsqueeze(-1))
        mean_new = mean_pt1.squeeze(-1).squeeze(-1) + mean_M
        variance_pt1 = torch.matmul(temp_term, covar_mM.unsqueeze(-1))
        variance_new = variance_M - variance_pt1.squeeze(-1).squeeze(-1)
        stdv_new = variance_new.clamp_min(CLAMP_LB).sqrt()

        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        view_shape = (
            [self.num_mv_samples] + [1] * (len(X.shape) - 2) + [self.num_fantasies]
        )

        max_vals = self.posterior_max_values.view(view_shape).unsqueeze(1)
        normalized_mvs_new = (max_vals - mean_new) / stdv_new
        cdf_mvs_new = normal.cdf(normalized_mvs_new).clamp_min(CLAMP_LB)

        stdv_M = variance_M.sqrt()
        normalized_mvs = (max_vals - mean_M) / stdv_M
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)

        log_pdf_fm = posterior_m.mvn.log_prob(samples_m).unsqueeze(0)
        H0 = posterior_m.mvn.entropy()

        Z = cdf_mvs_new / cdf_mvs
        h1 = -Z * Z.log() - Z * log_pdf_fm
        check_no_nans(h1)
        dim = [0, 1]
        H1_bar = h1.mean(dim=dim)
        h0 = -log_pdf_fm
        H0_bar = h0.mean(dim=dim)
        cov = ((h1 - H1_bar) * (h0 - H0_bar)).mean(dim=dim)
        beta = cov / (h0.var(dim=dim) * h1.var(dim=dim)).sqrt()
        H1_hat = H1_bar - beta * (H0_bar - H0)
        ig = H0 - H1_hat
        ig = ig.permute(-1, *range(ig.dim() - 1))
        return ig

    def _sample_max_values(self):

        with torch.no_grad():
            # sample max values
            self.posterior_max_values = _sample_max_value_Thompson(
                self.gp, self.candidate_set, self.num_mv_samples
            )

def _sample_max_value_Thompson(model, candidate_set, num_samples):
    """Samples the max values by discrete Thompson sampling.
    """
    posterior = model.posterior(candidate_set)
    samples = posterior.rsample(torch.Size([num_samples])).squeeze(-1)
    # samples is num_samples x (num_fantasies) x n
    max_values, _ = samples.max(dim=-1)
    if len(samples.shape) == 2:
        max_values = max_values.unsqueeze(-1)  # num_samples x num_fantasies

    return max_values