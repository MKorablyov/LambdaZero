import random, string, time, os, os.path as osp
import numpy as np
import pandas as pd
import ray, wandb
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Draw
import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


class Proxy:
    def __init__(self, update_freq, oracle, oracle_config, proposed_x, proposed_d, proposed_acq, logger):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_d, self.proposed_acq = proposed_x, proposed_d, proposed_acq
        self.logger = logger

        self.oracle = oracle(**oracle_config,logger=self.logger)
        self.num_acquisitions = 0

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

    def __call__(self, x, d, acq, info, y, v, num_acquisitions, logger):
        smiles, qed, synth_score = [xi["smiles"] for xi in x], [xi["qed"] for xi in x], [xi["synth_score"] for xi in x]
        df = pd.DataFrame({"smiles": smiles, "qed": qed, "synth_score": synth_score, "norm_dockscore": y,
                           "discount": d, "disc_dockscore":v})
        fname = str(num_acquisitions).zfill(5) + "_" + \
               ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        df.to_feather(os.path.join(self.outpath, fname + ".feather"))
        #print(pd.read_feather(os.path.join(self.outpath, oname + ".feather")))
        return None



class LogTrajectories:
    def __init__(self, max_steps, log_freq):
        self.max_steps = max_steps
        self.log_freq = log_freq

    def __call__(self, x, d, acq, info, y, v, num_acquisitions, logger):
        if num_acquisitions % self.log_freq == 0:
            trajs_smi = [xi["traj_smi"] for xi in x]
            mols = []
            for i in range(len(trajs_smi)):
                for j in range(self.max_steps):
                    if j < len(trajs_smi[i]):
                        mols.append(Chem.MolFromSmiles(trajs_smi[i][j]))
                    else:
                        mols.append(Chem.MolFromSmiles("H"))
            img = Draw.MolsToGridImage(mols, molsPerRow=self.max_steps, subImgSize=(250, 250), )
            logger.log_wandb_object.remote("acquired_traj_img", img, "image")

class LogTopKMols:
    def __init__(self, k, log_freq):
        self.k = k
        self.log_freq = log_freq
        self.best_k = pd.DataFrame({})

    def __call__(self, x, d, acq, info, y, v, num_acquisitions, logger):
        smiles, qed, synth_score = [xi["smiles"] for xi in x], [xi["qed"] for xi in x], [xi["synth_score"] for xi in x]
        df = pd.DataFrame({"smiles": smiles, "qed": qed, "synth_score": synth_score, "norm_dockscore": y,
                           "discount": d, "disc_dockscore": v})
        self.best_k = pd.concat([self.best_k, df])
        best_v = self.best_k["disc_dockscore"].to_numpy()
        best_v_idx = np.argsort(best_v)[-self.k:]  # highest values
        self.best_k = self.best_k.take(best_v_idx)
        if num_acquisitions % self.log_freq == 0:
            logger.log_wandb_object.remote("best_k_mols", self.best_k, "table")