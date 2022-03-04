import time
import torch
from copy import deepcopy
import os
import pickle
import gzip
import numpy as np
from argparse import Namespace

from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator, TrainBatch
from LambdaZero.examples.gflow.utils import LogMetrics, TimeStats
from LambdaZero.examples.gflow.trainer.basic_trainer import BasicTrainer


_stop = [None]


class TrainGFlowTrajBal(BasicTrainer):
    def __init__(self,
                 args,
                 model=None, proxy=None, dataset: DataGenerator=None, do_save=True):

        super().__init__(args, model, proxy, dataset, do_save)

    def train_epoch(self, epoch: int, train_batch: TrainBatch):
        dataset = self.dataset
        model = self.model
        device = self.device
        t_metrics = self._train_metrics
        opt = self.optimizer
        args = self.args
        do_save = self.do_save

        last_losses = []
        if self._train_start_time is None:
            self._train_start_time = time.time()

        _t_losses_k, _t_infos_k = self._train_losses_k, self._train_infos_k

        tf = lambda x: torch.tensor(x, device=device).to(self.float_ttype)
        clip_loss = tf([args.clip_loss])

        b_p, b_pb, b_a, b_r, b_s, b_d, mols = train_batch
        idc, n = train_batch.idx, train_batch.n
        ntransitions = b_r.shape[0]

        self.train_num_mol += ntransitions

        # state outputs
        stem_out_s, mol_out_s = model(b_p)
        logits = model.action_neg_log_likelihood(b_p, b_a, 0, stem_out_s, mol_out_s)
        tzeros = torch.zeros(idc[-1]+1, device=device, dtype=args.floatX)
        traj_logits = tzeros.index_add(0, idc, logits)
        traj_r = tzeros.index_add(0, idc, b_r)
        uniform_log_PB = tzeros.index_add(0, idc, torch.log(1/n))
        losses = ((model.logZ + traj_logits) - (torch.log(traj_r) + uniform_log_PB)).pow(2)
        loss = losses.mean()
        
        opt.zero_grad()
        loss.backward(retain_graph=(not epoch % 50))


        last_losses.append((loss.item()))

        t_metrics.update(
            _t_losses_k,
            [loss.item()]
        )

        if not epoch % 50:
            """ 
                add some metrics to log (just _log_metrics of them will be logged to wandb though) 
            """
            t_metrics.update(_t_infos_k, [
                # _term_loss.data.cpu().numpy(),
                # _flow_loss.data.cpu().numpy(),
                # exp_inflow.data.cpu().numpy(),
                # exp_outflow.data.cpu().numpy(),
                b_r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                # torch.autograd.grad(loss, qsa_p, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_s, retain_graph=True)[0].data.cpu().numpy(),
                # torch.autograd.grad(loss, stem_out_p, retain_graph=True)[0].data.cpu().numpy(),
                self.train_num_mol / (time.time() - self._train_start_time)
            ])
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

        opt.step()

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
            "train_num_mol": self.train_num_mol,
            "total_num_steps": self.train_num_mol,
        })

        if do_save:
            self.save_stuff()

        return {"losses": losses.detach()}, log_metrics

