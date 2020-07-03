import numpy as np
import torch
from torch import nn
from torch import functional as F

class TrainableRandomDistribution(nn.Module):
    # Samples weights for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)
    # Calculates the variational posterior part of the complexity part of the loss
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi
        # self.normal = torch.distributions.Normal(0, 1)

    def sample(self):
        """ Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is
        a function from a trainable parameter, and adding a mean
        sets those weights as the current ones
        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w

    def log_posterior(self, w=None):
        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost
        returns:
            torch.tensor with shape []
        """

        assert (self.w is not None), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w

        log_sqrt2pi = np.log(np.sqrt(2 * self.pi))
        log_posteriors = -log_sqrt2pi - torch.log(self.sigma) - (((w - self.mu) ** 2) / (2 * self.sigma ** 2)) - 0.5
        return log_posteriors.sum()


class PriorWeightDistribution(nn.Module):
    # Calculates a Scale Mixture Prior distribution for the prior part of the complexity cost on Bayes by Backprop paper
    def __init__(self,
                 pi=1,
                 sigma1=0.1,
                 sigma2=0.001,
                 dist=None):
        super().__init__()

        if (dist is None):
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)

        if (dist is not None):
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None


class BayesianModule(nn.Module):
    """
    creates base class for BNN, in order to enable specific behavior
    """
    def init(self):
        super().__init__()

class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).
    Its objective is be interactable with torch nn.Module API,
    being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1=0.1,
                 prior_sigma_2=0.4,
                 prior_pi=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-7.0,
                 freeze=False,
                 prior_dist=None):
        super().__init__()

        # our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        # Variational weight parameters and sample
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # Variational bias parameters and sample
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2,
                                                         dist=self.prior_dist)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2,
                                                       dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Sample the weights and forward it

        # if the model is frozen, return frozen
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features))
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.linear(x, w, b)

    def forward_frozen(self, x):
        """
        Computes the feedforward operation with the expected value for weight and biases
        """
        if self.bias:
            return F.linear(x, self.weight_mu, self.bias_mu)
        else:
            return F.linear(x, self.weight_mu, torch.zeros(self.out_features))