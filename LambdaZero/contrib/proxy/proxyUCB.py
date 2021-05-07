import random

import numpy as np
import ray
from LambdaZero.contrib.acquisition_function import UCB
from .proxy import Proxy
import torch
from sklearn.metrics import explained_variance_score
import os

@ray.remote(num_gpus=0.3, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self, update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config, logger):
        # load data for (1) acquisition function (2) proxy
        self.seen_x, self.seen_y, self.val_x, self.val_y = load_seen(**load_seen_config)
        proposed_x, proposed_d, proposed_acq = [], [], []

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger)
        self.UCB = UCB(**acquirer_config)
        self.UCB.model.fit(self.seen_x, self.seen_y)
        self.oracle = oracle(**oracle_config)

    def acquire_and_update(self, proposed_x, proposed_d, proposed_acq):
        # if self.acquired_batches < self.init_random_batches:
        x, d, acq, info = self.UCB.acquire_batch(proposed_x, proposed_d, proposed_acq)
        y = self.oracle(x)

        # ==========================================================================================
        # DEBUG - calculate oracle and network prediction for unseed values
        # x_unacquired = list(np.random.choice(info["x_unacquired"],
        #                                      min(50, len(info["x_unacquired"])), replace=True))
        # y_unacquired = self.oracle(x_unacquired)
        # before_mean, before_var = self.UCB.model.get_mean_and_variance(x_unacquired)
        # unacq_before_expv = explained_variance_score(y_unacquired, before_mean)

        # ==========================================================================================

        cand_stats = self.post_acquire_and_update(x, y)
        v = np.asarray(y) * np.asarray(d)

        self.logger.log.remote([{
            "proxy/proposed_acq_mean": np.mean(proposed_acq),
            "proxy/proposed_acq_max": np.max(proposed_acq),
            "proxy/proposed_acq_min": np.min(proposed_acq),

            "proxy/acquired_acq_mean": np.mean(acq),
            "proxy/acquired_acq_max": np.max(acq),
            "proxy/acquired_acq_min": np.min(acq),

            "proxy/acquired_v_mean": np.mean(v),
            "proxy/acquired_v_max": np.max(v),
            "proxy/acquired_v_min": np.min(v),

            **cand_stats,
        }])

        self.UCB.update_with_seen(self.seen_x, self.seen_y, x, y)

        # ==========================================================================================
        # DEBUG - calculate oracle and network prediction for unseed values
        # after_mean, after_var = self.UCB.model.get_mean_and_variance(x_unacquired)
        # unacq_after_expv = explained_variance_score(y_unacquired, after_mean)
        #
        # unacquired = dict({
        #     "unacq_before_mean": before_mean,
        #     "unacq_before_var": before_var,
        #     "unacq_target": y_unacquired,
        #     "unacq_after_mean": after_mean,
        #     "unacq_after_var": after_var,
        #     "unacq": x_unacquired,
        #     "acq": x,
        #     "unacq_before_expv": unacq_before_expv,
        #     "unacq_after_expv": unacq_after_expv,
        # })
        # cfg_name = ray.get(self.logger.config_name.remote())
        # out_path = f"/home/andrein/scratch/Summaries/debug/{cfg_name}_unacq_model_pred_"
        # print(out_path)
        # torch.save(unacquired, f"{out_path}_{self.UCB.model._istep-1}")
        # print("SAVE unacquired------")
        # ==========================================================================================

        self.seen_x, self.seen_y = self.seen_x + x, self.seen_y + y
        return None

    def get_acquisition_func(self):
        return self.UCB