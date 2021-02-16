import time
import numpy as np
from .acquisition_function import AcquisitionFunction
from torch.distributions import Normal

class EI(AcquisitionFunction):
    def __init__(self, model, model_config, acq_size):
        self.model = model(**model_config)
        AcquisitionFunction.__init__(self, model, model_config, acq_size)

    def update_with_seen(self, x, y):
        self.model.fit(x,y)
        # todo: evaluate acquisition function on a separate train/val set
        return None

    def acquisition_value(self, x):
        posterior = self.model.posterior(X.unsqueeze(-3))
        mean = self.weight * posterior.mean.squeeze(-1).squeeze(-1)
        variance = posterior.variance.clamp_min(CLAMP_LB).view_as(mean)
        check_no_nans(mean)
        check_no_nans(variance)

        ig = self._compute_information_gain(
            X=X, mean_M=mean, variance_M=variance, covar_mM=variance.unsqueeze(-1)
        )
        return ig.mean(dim=0)

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
        posterior_m = self.model.posterior(X.unsqueeze(-3), observation_noise=True)
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
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
        if self.X_pending is None:
            view_shape[-1] = 1
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

def _sample_max_value_Thompson(model, candidate_set, num_samples, maximize):
    """Samples the max values by discrete Thompson sampling.
    """
    posterior = model.posterior(candidate_set)
    weight = 1.0 if maximize else -1.0
    samples = weight * posterior.rsample(torch.Size([num_samples])).squeeze(-1)
    # samples is num_samples x (num_fantasies) x n
    max_values, _ = samples.max(dim=-1)
    if len(samples.shape) == 2:
        max_values = max_values.unsqueeze(-1)  # num_samples x num_fantasies

    return max_values