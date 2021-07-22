import time
import torch
from copy import deepcopy
import os
import pickle
import gzip
import numpy as np
from argparse import Namespace

from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator
from LambdaZero.examples.gflow.utils import LogMetrics, TimeStats
from LambdaZero.examples.gflow.trainer.trainer_v2 import TrainGFlowV2


_stop = [None]


class TrainGFlowFwdBack(TrainGFlowV2):
    def __init__(self,
                 args,
                 model=None, proxy=None, dataset: DataGenerator=None, do_save=True):

        super().__init__(args, model, proxy, dataset, do_save)

    def train_epoch(self, epoch: int):
        debug_no_threads = self.debug_no_threads
        dataset = self.dataset
        model = self.model
        device = self.device
        t_metrics = self._train_metrics
        opt = self.optimizer
        args = self.args
        do_save = self.do_save

        loginf = 1000  # to prevent nans
        last_losses = []
        if self._train_start_time is None:
            self._train_start_time = time.time()

        _t_losses_k, _t_infos_k = self._train_losses_k, self._train_infos_k

        tf = lambda x: torch.tensor(x, device=device).to(torch.float64 if args.floatX == 'float64' else torch.float32)
        clip_loss = tf([args.clip_loss])

        self._ts_stats.sample.start()
        p, pb, a, r, s, d, mols = dataset.sample_train_batch(args.batch_size)
        self._ts_stats.sample.end()
        self._ts_stats.train.start()

        ntransitions = r.shape[0]

        self.train_num_mol += ntransitions

        self._ts_stats.outflow.start()
        # Calculate forward policy log prob and value
        p_stem_out, p_mol_out, _ = model.run_model(p)
        p_mol_out, p_v_out = p_mol_out[:, :1], p_mol_out[:, 1:]  # Assume two values out / molecule
        f_action_logprob = -model.action_negloglikelihood(p, a, 0, p_stem_out, p_mol_out)
        # Forward Q approximation based on V(p) * pi_fwd(a, p)
        fwd_flow = torch.log(args.log_reg_c + torch.exp(p_v_out + f_action_logprob))
        self._ts_stats.outflow.end()

        self._ts_stats.inflow.start()
        # Calculate backward policy log prob - based on delete per bond atom prediction
        s_stem_out, s_mol_out, s_jbond_out = model.run_model(s)
        s_mol_out, s_v_out = s_mol_out[:, :1], s_mol_out[:, 1:]
        bond_act = torch.zeros_like(a)
        b_action_logprob = -model.action_negloglikelihood_bonds(s, bond_act, s_jbond_out)
        # Backward Q approximation based on V(s) * pi_fwd(a, s)
        bck_flow = torch.log(args.log_reg_c + torch.exp(s_v_out + b_action_logprob))

        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        bck_flow_plus_r = torch.log(args.log_reg_c + r) + bck_flow * (1-d)
        self._ts_stats.inflow.end()

        self._ts_stats.lossbackward.start()
        if args.do_nblocks_reg:
            losses = _losses = ((fwd_flow - bck_flow_plus_r) / (s.nblocks * args.max_blocks)).pow(2)
        else:
            losses = _losses = (fwd_flow - bck_flow_plus_r).pow(2)

        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)

        term_loss = (losses * d).sum() / (d.sum() + 1e-20)
        flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        if args.balanced_loss:
            loss = term_loss * args.leaf_coef + flow_loss
        else:
            loss = losses.mean()
        opt.zero_grad()
        loss.backward(retain_graph=(not epoch % 50))
        self._ts_stats.lossbackward.end()

        self._ts_stats.optstep.start()
        _term_loss = (_losses * d).sum() / (d.sum() + 1e-20)
        _flow_loss = (_losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))

        t_metrics.update(
            _t_losses_k,
            [loss.item(), _term_loss.item(), _flow_loss.item(), term_loss.item(), flow_loss.item()]
        )

        if not epoch % 50:
            t_metrics.update(_t_infos_k, [
                _term_loss.data.cpu().numpy(),
                _flow_loss.data.cpu().numpy(),
                fwd_flow.data.cpu().numpy(),
                bck_flow.data.cpu().numpy(),
                r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                torch.autograd.grad(loss, f_action_logprob, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, b_action_logprob, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, fwd_flow, retain_graph=True)[0].data.cpu().numpy(),
                self.train_num_mol / (time.time() - self._train_start_time)
            ])
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

        opt.step()
        self._ts_stats.optstep.end()
        self._ts_stats.train.end()

        model.training_steps = epoch + 1
        if args.bootstrap_tau > 0:
            for _a,b in zip(model.parameters(), self.target_model.parameters()):
                b.data.mul_(1-args.bootstrap_tau).add_(args.bootstrap_tau*_a)

        if not epoch % 100:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(epoch, last_losses)
            print('time:', time.time() - self.time_last_check)
            self.time_last_check = time.time()
            last_losses = []

            for k, v in self._ts_stats.__dict__.items():
                print(f"{k}: {v.stats():.6f}")

            if not epoch % 1000 and do_save:
                self.save_stuff()

        log_metrics = t_metrics.get_stats(self._log_metrics)
        log_metrics.update({
            "train_num_sampled_mols": len(dataset.sampled_mols),
            "train_num_mol": self.train_num_mol,
            "total_num_steps": self.train_num_mol,
        })

        if do_save:
            self.save_stuff()

        return model, dataset, log_metrics
