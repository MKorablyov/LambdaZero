from ray.tune import Stopper
import numpy as np

"""
NOTE that this early stopping stopper is implemented in the `master` branch
of tune, but not in the `latest` (ie pip package) code. I copy-paste here,
but we should use the package version once it becomes available in the package.
"""


class EarlyStopping(Stopper):
    def __init__(self, metric, std=0.001, top=10, mode="min", patience=0):
        """Create the EarlyStopping object.

        Stops the entire experiment when the metric has plateaued
        for more than the given amount of iterations specified in
        the patience parameter.

        Args:
            metric (str): The metric to be monitored.
            std (float): The minimal standard deviation after which
                the tuning process has to stop.
            top (int): The number of best model to consider.
            mode (str): The mode to select the top results.
                Can either be "min" or "max".
            patience (int): Number of epochs to wait for
                a change in the top models.

        Raises:
            ValueError: If the mode parameter is not "min" nor "max".
            ValueError: If the top parameter is not an integer
                greater than 1.
            ValueError: If the standard deviation parameter is not
                a strictly positive float.
            ValueError: If the patience parameter is not
                a strictly positive integer.
        """
        if mode not in ("min", "max"):
            raise ValueError("The mode parameter can only be"
                             " either min or max.")
        if not isinstance(top, int) or top <= 1:
            raise ValueError("Top results to consider must be"
                             " a positive integer greater than one.")
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("Patience must be"
                             " a strictly positive integer.")
        if not isinstance(std, float) or std <= 0:
            raise ValueError("The standard deviation must be"
                             " a strictly positive float number.")
        self._mode = mode
        self._metric = metric
        self._patience = patience
        self._iterations = 0
        self._std = std
        self._top = top
        self._top_values = []

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        self._top_values.append(result[self._metric])
        if self._mode == "min":
            self._top_values = sorted(self._top_values)[:self._top]
        else:
            self._top_values = sorted(self._top_values)[-self._top:]

        # If the current iteration has to stop
        if self.has_plateaued():
            # we increment the total counter of iterations
            self._iterations += 1
        else:
            # otherwise we reset the counter
            self._iterations = 0

        # and then call the method that re-executes
        # the checks, including the iterations.
        return self.stop_all()

    def has_plateaued(self):
        return (len(self._top_values) == self._top
                and np.std(self._top_values) <= self._std)

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return self.has_plateaued() and self._iterations >= self._patience