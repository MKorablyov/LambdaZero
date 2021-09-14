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


class TrainGFlowFwdBackPPO(BasicTrainer):
    def __init__(self,
                 args,
                 model=None, proxy=None, dataset: DataGenerator=None, do_save=True):

        super().__init__(args, model, proxy, dataset, do_save)
        self._inner_epochs = getattr(args, "inner_epochs", 4)
        self._clip_ppo = getattr(args, "clip_ppo", 0.1)

    def train_epoch(self, epoch: int, train_batch: TrainBatch):
        debug_no_threads = self.debug_no_threads
        dataset = self.dataset
        model = self.model
        device = self.device
        t_metrics = self._train_metrics
        opt = self.optimizer
        args = self.args
        do_save = self.do_save
        clip_policy = args.clip_policy

        loginf = 1000  # to prevent nans
        last_losses = []
        if self._train_start_time is None:
            self._train_start_time = time.time()

        _t_losses_k, _t_infos_k = self._train_losses_k, self._train_infos_k

        tf = lambda x: torch.tensor(x, device=device).to(self.float_ttype)
        clip_loss = tf([args.clip_loss])
        log_log_reg_c = torch.log(tf([args.log_reg_c])[0])

        b_p, b_pb, b_a, b_r, b_s, b_d, mols = train_batch

        ntransitions = b_r.shape[0]

        self.train_num_mol += ntransitions
        init_values = None

        for inner in range(self._inner_epochs):
            # Calculate forward / backward policy log probs and values
            p_stem_out, p_mol_out, _ = model.run_model(b_p.clone())
            s_stem_out, s_mol_out, s_jbond_out = model.run_model(b_s.clone())
            s_mol_out, s_v_out = s_mol_out[:, :1], s_mol_out[:, 1]
            p_mol_out, p_v_out = p_mol_out[:, :1], p_mol_out[:, 1]  # Assume two values out / molecule

            if clip_policy > 0:
                for ix, p_out in enumerate([p_stem_out, p_mol_out, s_stem_out, s_mol_out, s_jbond_out]):
                    p_out.clip_(-clip_policy, clip_policy)

            # Forward Q approximation based on V(p) * pi_fwd(a, p)
            f_action_logprob = -model.action_negloglikelihood(b_p, b_a, 0, p_stem_out, p_mol_out)

            if init_values is None:
                init_values = f_action_logprob.clone()
            elif self._clip_ppo > 0.:
                ratio = torch.exp(f_action_logprob - init_values)
                stop_g = torch.abs(ratio - 1) > self._clip_ppo
                # stop_g.fill_(True)
                num_stop = stop_g.sum().item()
                # print(num_stop, f_action_logprob.size())
                if num_stop > 0:
                    f_action_logprob[stop_g] = f_action_logprob[stop_g].detach()
                # surr1 = ratio * adv_targ
                # surr2 = torch.clamp(ratio, 1.0 - self._clip_ppo, 1.0 + self._clip_ppo) #  * adv_targ
                # action_loss = -torch.min(surr1, surr2).mean()


            fwd_flow = torch.logaddexp(log_log_reg_c, p_v_out + f_action_logprob)

            # Calculate backward policy log prob - based on delete per bond atom prediction
            bond_act = torch.zeros_like(b_a)
            # b_action_logprob = -model.action_negloglikelihood_bonds(s, bond_act, s_jbond_out)

            b_cnt = torch.tensor(b_s.__slices__['bonds'])
            pjbonds = (b_cnt[1:] - b_cnt[:-1]).clamp(min=1)
            b_action_logprob = torch.log(1 / pjbonds).to(p_stem_out.dtype).to(p_stem_out.device)

            # Backward Q approximation based on V(s) * pi_fwd(a, s)
            bck_flow = torch.exp(s_v_out + b_action_logprob)

            # include reward and done multiplier, then take the log
            # we're guarenteed that r > 0 iff d = 1, so the log always works
            bck_flow_plus_r = torch.log(args.log_reg_c + b_r + bck_flow * (1-b_d))

            if args.do_nblocks_reg:
                losses = _losses = ((fwd_flow - bck_flow_plus_r) / (b_s.nblocks * args.max_blocks)).pow(2)
            else:
                losses = _losses = (fwd_flow - bck_flow_plus_r).pow(2)

            if clip_loss > 0:
                ld = losses.detach()
                losses = losses / ld * torch.minimum(ld, clip_loss)

            term_loss = (losses * b_d).sum() / (b_d.sum() + 1e-20)
            flow_loss = (losses * (1-b_d)).sum() / ((1-b_d).sum() + 1e-20)

            if args.balanced_loss:
                loss = term_loss * args.leaf_coef + flow_loss
            else:
                loss = losses.mean()

            opt.zero_grad()
            loss.backward(retain_graph=(not epoch % 50))

            _term_loss = (_losses * b_d).sum() / (b_d.sum() + 1e-20)
            _flow_loss = (_losses * (1-b_d)).sum() / ((1-b_d).sum() + 1e-20)
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
                    b_r.data.cpu().numpy(),
                    mols[1],
                    [i.pow(2).sum().item() for i in model.parameters()],
                    torch.autograd.grad(loss, f_action_logprob, retain_graph=True)[0].data.cpu().numpy(),
                    torch.autograd.grad(loss, f_action_logprob, retain_graph=True)[0].data.cpu().numpy(),
                    torch.autograd.grad(loss, fwd_flow, retain_graph=True)[0].data.cpu().numpy(),
                    self.train_num_mol / (time.time() - self._train_start_time)
                ])
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

            opt.step()

            model.training_steps = epoch + 1
            if args.bootstrap_tau > 0:
                for _a,b in zip(model.parameters(), self.target_model.parameters()):
                    b.data.mul_(1-args.bootstrap_tau).add_(args.bootstrap_tau*_a)

        if not epoch % 50:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(epoch, last_losses)
            print('time:', time.time() - self.time_last_check)
            self.time_last_check = time.time()
            last_losses = []

            if not epoch % 1000 and do_save:
                self.save_stuff()

        log_metrics = t_metrics.get_stats(self._log_metrics)

        log_metrics.update({
            "train_num_mol": self.train_num_mol,
            "total_num_steps": self.train_num_mol,
        })

        if do_save:
            self.save_stuff()

        return {"losses": _losses.detach()}, log_metrics

