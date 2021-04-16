import numpy as np
import ray
from LambdaZero.contrib.acquisition_function import UCB
from .proxy import Proxy
import torch

@ray.remote(num_gpus=0.3, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self, update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config, logger):
        # load data for (1) acquisition function (2) proxy
        self.seen_x, self.seen_y, self.val_x, self.val_y = load_seen(**load_seen_config)
        proposed_x, proposed_d, proposed_acq = [], [], []  # todo: load if needed

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger)
        self.UCB = UCB(**acquirer_config)
        self.UCB.model.fit(self.seen_x, self.seen_y)
        self.oracle = oracle(**oracle_config)

    def acquire_and_update(self, proposed_x, proposed_d, proposed_acq):
        x, d, acq, info = self.UCB.acquire_batch(proposed_x, proposed_d, proposed_acq)
        y = self.oracle(x)

        # ==========================================================================================
        # DEBUG - calculate oracle and network prediction for unseed values
        x_unacquired = list(np.random.choice(info["x_unacquired"],
                                             min(50, len(info["x_unacquired"])), replace=True))
        y_unacquired = self.oracle(x_unacquired)
        before_mean, before_var = self.UCB.model.get_mean_and_variance(x_unacquired)
        # ==========================================================================================

        cand_stats = self.post_acquire_and_update(x, y)

        self.logger.log.remote([{
            "proxy/proposed_acq_mean": np.mean(proposed_acq),
            "proxy/proposed_acq_max": np.max(proposed_acq),
            "proxy/proposed_acq_min": np.min(proposed_acq),

            "proxy/acquired_acq_mean": np.mean(acq),
            "proxy/acquired_acq_max": np.max(acq),
            "proxy/acquired_acq_min": np.min(acq),

            "proxy/acquired_y_mean": np.mean(y),
            "proxy/acquired_y_max": np.max(y),
            "proxy/acquired_y_min": np.min(y),
            **cand_stats
        }])

        self.UCB.update_with_seen(self.seen_x, self.seen_y, x, y)

        # ==========================================================================================
        # DEBUG - calculate oracle and network prediction for unseed values
        after_mean, after_var = self.UCB.model.get_mean_and_variance(x_unacquired)
        unacquired = dict({
            "unacq_before_mean": before_mean,
            "unacq_before_var": before_var,
            "unacq_target": y_unacquired,
            "unacq_after_mean": after_mean,
            "unacq_after_var": after_var,
            "unacq": x_unacquired,
            "acq": x,
        })
        print(out_path)
        out_path = "/home/andrein/scratch/Summaries/debug/30k_unacq_model_pred_"
        torch.save(unacquired, f"{out_path}_{self.UCB.model._istep-1}")
        print("SAVE unacquired------")
        # ==========================================================================================

        return None

    def get_acquisition_func(self):
        return self.UCB
