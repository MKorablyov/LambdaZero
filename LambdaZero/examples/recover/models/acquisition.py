import torch
from torch.nn.functional import relu


########################################################################################################################
# Abstract Acquisition
########################################################################################################################


class AbstractAcquisition:
    def __init__(self, config):
        pass

    def get_scores(self, output):
        raise NotImplementedError

    def update_with_seen(self, seen_labels):
        raise NotImplementedError


########################################################################################################################
# Acquisition functions
########################################################################################################################


class ExpectedImprovement(AbstractAcquisition):
    def __init__(self, config):
        super().__init__(config)
        self.s_max = 0.

    def get_scores(self, output):
        scores = relu(output - self.s_max)
        return scores.sum(dim=1).to("cpu")

    def update_with_seen(self, seen_labels):
        self.s_max = max(self.s_max, torch.max(seen_labels).item())


class RandomAcquisition(AbstractAcquisition):
    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        return torch.randn(output.shape[0])

    def update_with_seen(self, seen_labels):
        pass


class GPUCB(AbstractAcquisition):
    def __init__(self, config):
        super().__init__(config)
        # TODO: have a kappa that gets smaller along training
        self.kappa = config["kappa"]

    def get_scores(self, output):
        mean = output.mean(dim=1)
        std = output.std(dim=1)

        scores = mean + self.kappa * std

        return scores.to("cpu")

    def update_with_seen(self, seen_labels):
        pass


class GreedyAcquisition(AbstractAcquisition):
    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        scores = output.mean(dim=1).to("cpu")

        return scores

    def update_with_seen(self, seen_labels):
        pass


class GaussianThompson(AbstractAcquisition):
    """
    Thompson sampling assuming that the posterior distribution of the targets are Gaussian
    """
    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        # We estimate the parameters of the posterior p(y given x) assumed to be gaussian
        mean = output.mean(dim=1)
        std = output.std(dim=1)

        # We sample from these posterior
        eps = torch.randn_like(std)
        scores = mean + eps * std

        return scores

    def update_with_seen(self, seen_labels):
        pass


class Thompson(AbstractAcquisition):
    """
    Thompson sampling without Gaussian assumption. When using this acquisition function, it is better to use:
    - small batch size
    - high n_scoring_forward_passes
    as the scores within a batch are correlated (same dropout configurations were used)
    """
    def __init__(self, config):
        super().__init__(config)

    def get_scores(self, output):
        # Select one sample at random for each example in the batch
        idx = torch.randint(high=output.shape[1], size=(output.shape[0],))[:, None]
        scores = output.gather(1, idx)[:, 0]

        return scores

    def update_with_seen(self, seen_labels):
        pass