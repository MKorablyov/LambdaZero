import random, string, time, os, os.path as osp
import pandas as pd
import ray
from copy import deepcopy

import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


class Proxy:
    def __init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_d, self.proposed_acq = proposed_x, proposed_d, proposed_acq
        self.logger = logger

    def propose_x(self,x, d, acq):
        """
        :param x: points to evaluate
        :param aq: current estimates of the acquisition value at x
        :param d: discount factors defaults to 1. This would be used for the acquisition
        :return:
        """
        self.proposed_x.extend(x)
        self.proposed_d.extend(d)
        self.proposed_acq.extend(acq)
        # update model and acquisition function if needed
        if len(self.proposed_x) >= self.update_freq:
            # todo: a better solution would be to
            #  keep and re-compute acqusition values of proposed while proxy-model was retraining
            #proposed_x, proposed_d, proposed_acq = self.proposed_x, self.proposed_d, self.proposed_acq
            #self.proposed_x, self.proposed_d, self.proposed_acq = [], [], []
            self.acquire_and_update(self.proposed_x, self.proposed_d, self.proposed_acq)
            self.proposed_x, self.proposed_d, self.proposed_acq = [], [], []
        return None

    def acquire_and_update(self, proposed_x, proposed_d, proposed_acq):
        raise NotImplementedError

    def acquire(self, x):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError


class Actor():
    def __init__(self, scoreProxy, actor_sync_freq):
        self.scoreProxy = scoreProxy
        self.sync_freq = actor_sync_freq
        # initialize
        self.num_calls = 0
        self.acquisition_func = ray.get(scoreProxy.get_acquisition_func.remote())
        self.names = []

    def __call__(self, x, d):
        # compute acquisition value
        self.num_calls += 1
        acq, info = self.acquisition_func.acquisition_value(x)

        # send molecule to the remote proxy
        self.scoreProxy.propose_x.remote(deepcopy(x), deepcopy(d), deepcopy(acq))

        # sync weights with proxy if needed
        if self.num_calls % self.sync_freq==1:
            # todo - need to figure out to do non-blocking calls here
            self.acquisition_func = ray.get(self.scoreProxy.get_acquisition_func.remote())
        return acq, info


class SaveDocked:
    def __init__(self, outpath=osp.join(summaries_dir, "save_docked")):
        self.outpath = outpath
        if not osp.exists(outpath):os.makedirs(outpath)

    def __call__(self, x, d, acq, info, y, name=None):
        name = name or ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        smiles, qed, synth_score = [xi["smiles"] for xi in x], [xi["qed"] for xi in x], [xi["synth_score"] for xi in x]
        df = pd.DataFrame(
            {"smiles": smiles, "qed": qed, "synth_score": synth_score, "norm_dockscore": y, "discount": d})
        df.to_feather(os.path.join(self.outpath, name + ".feather"))
        #print(pd.read_feather(os.path.join(self.outpath, oname + ".feather")))
        return None