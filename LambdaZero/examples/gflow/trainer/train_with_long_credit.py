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


class TrainLongCredit(BasicTrainer):
    def __init__(self,
                 args,
                 model=None, proxy=None, dataset: DataGenerator=None, do_save=True):

        super().__init__(args, model, proxy, dataset, do_save)

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

        self._ts_stats.sample.start()
        b_p, b_pb, b_a, b_r, b_s, b_d, mols = train_batch
        self._ts_stats.sample.end()

        self._ts_stats.train.start()

        ntransitions = b_r.shape[0]

        self.train_num_mol += ntransitions

        self._ts_stats.inflow.start()
        # state outputs
        if args.bootstrap_tau > 0:
            with torch.no_grad():
                stem_out_s, mol_out_s = self.target_model(b_s)
        else:
            stem_out_s, mol_out_s = model(b_s)
        # parents of the state outputs
        stem_out_p, mol_out_p = model(b_p)

        if clip_policy > 0:
            for ix, p_out in enumerate([stem_out_s, mol_out_s, stem_out_p, mol_out_p]):
                p_out.clip_(-clip_policy, clip_policy)

        # index parents by their corresponding actions
        qsa_p = model.index_output_by_action(b_p, stem_out_p, mol_out_p[:, 0], b_a)

        # then sum the parents' contribution, this is the inflow
        exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                      .index_add_(0, b_pb, torch.exp(qsa_p))) # pb is the parents' batch index
        inflow = torch.log(exp_inflow + args.log_reg_c)
        self._ts_stats.inflow.end()

        self._ts_stats.outflow.start()
        # sum the state's Q(s,a), this is the outflow
        exp_outflow = model.sum_output(b_s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(args.log_reg_c + b_r + exp_outflow * (1-b_d))
        if args.do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) / (b_s.nblocks * args.max_blocks)).pow(2)
        else:
            losses = _losses = (inflow - outflow_plus_r).pow(2)
        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)
        self._ts_stats.outflow.end()

        self._ts_stats.lossbackward.start()
        term_loss = (losses * b_d).sum() / (b_d.sum() + 1e-20)
        flow_loss = (losses * (1-b_d)).sum() / ((1-b_d).sum() + 1e-20)
        priority = _losses.detach().clone()
        if args.balanced_loss:
            loss = term_loss * args.leaf_coef + flow_loss
            priority[b_d == 1] *= args.leaf_coef
        else:
            loss = losses.mean()

        # ==========================================================================================
        # add extra flow
        # Assume ordered states by trajectory sample
        # import pdb; pdb.set_trace()

        terminal_states = torch.where(b_d)[0].long()
        terminal_states = terminal_states[terminal_states > 0]  # correct for horizon in traj
        # outflow_terminal = torch.log(args.log_reg_c + b_r[terminal_states])

        # Calculate Q(st,a-at)
        p_mols, s_mols = mols
        action_idx = []
        prev_term = 0
        extra_loss = []
        for term_idx in terminal_states:
            prev_state = term_idx - 1
            prev_act = s_mols[prev_state].action
            exp_outflow_prime = exp_outflow[prev_state]
            prev_inflow = inflow[prev_state]

            # Remove Q(st,at)
            if prev_act == 0:
                exp_outflow_prime -= torch.exp(mol_out_s[prev_state, 0])
            else:
                stem_idx = torch.where(b_s.stems_batch == prev_state)[0][prev_act[1]]
                exp_outflow_prime -= torch.exp(stem_out_s[stem_idx, prev_act[0]])

            # Determine Q(st+1, inflow - at) Must see which action
            found_mol = None
            tgt_smi = s_mols[prev_state].smiles
            for ipmol, p_mol in enumerate(p_mols[term_idx]):
                if p_mol.smiles == tgt_smi:
                    found_mol = ipmol
                    break

            exp_inflow_prime = exp_inflow[term_idx]

            # index parents by their corresponding actions
            # qsa_p = model.index_output_by_action(b_p, stem_out_p, mol_out_p[:, 0], b_a)
            p_idx = torch.where(b_pb == term_idx)[0][found_mol]
            exp_inflow_prime -= torch.exp(qsa_p[p_idx])

            # add reward and everythng
            optim_val = args.log_reg_c + b_r[term_idx] + exp_outflow_prime - exp_inflow_prime
            if optim_val >= 0:
                outflow_lc_plus_r = torch.log(optim_val)
                extra_loss.append((prev_inflow - outflow_lc_plus_r).pow(2))
            else:
                print("SMALLER :(", optim_val)

        # import pdb; pdb.set_trace()

        loss += sum(extra_loss)/len(extra_loss)
        # ==========================================================================================

        # ==========================
        opt.zero_grad()
        loss.backward(retain_graph=(not epoch % 50))
        self._ts_stats.lossbackward.end()

        self._ts_stats.optstep.start()
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
                exp_inflow.data.cpu().numpy(),
                exp_outflow.data.cpu().numpy(),
                b_r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                torch.autograd.grad(loss, qsa_p, allow_unused=True, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_s, allow_unused=True, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_p, allow_unused=True, retain_graph=True)[0].data.cpu().numpy(),
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
            "train_num_mol": self.train_num_mol,
            "total_num_steps": self.train_num_mol,
        })

        if do_save:
            self.save_stuff()

        return {"losses": priority.detach()}, log_metrics
