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


class GflowCreditAssign(BasicTrainer):
    def __init__(self,
                 args,
                 model=None, proxy=None, dataset: DataGenerator=None, do_save=True):

        super().__init__(args, model, proxy, dataset, do_save, extra_t_info=["credit_loss"])

    def train_epoch(self, epoch: int, train_batch: TrainBatch):
        dataset = self.dataset
        model = self.model
        device = self.device
        t_metrics = self._train_metrics
        opt = self.optimizer
        args = self.args
        do_save = self.do_save
        clip_policy = args.clip_policy
        t_correction_coef = getattr(args, "t_correction_coef", 0.01)
        t_avg_by_trajectory = getattr(args, "t_avg_by_trajectory", False)

        loginf = 1000  # to prevent nans
        last_losses = []
        if self._train_start_time is None:
            self._train_start_time = time.time()

        _t_losses_k, _t_infos_k = self._train_losses_k, self._train_infos_k

        tf = lambda x: torch.tensor(x, device=device).to(self.float_ttype)
        clip_loss = tf([args.clip_loss])

        b_p, b_pb, b_a, b_r, b_s, b_d, mols = train_batch

        ntransitions = b_r.shape[0]

        self.train_num_mol += ntransitions

        # state outputs
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

        # sum the state's Q(s,a), this is the outflow
        exp_outflow = model.sum_output(b_s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(args.log_reg_c + b_r + exp_outflow * (1-b_d))

        losses = _losses = (inflow - outflow_plus_r).pow(2)

        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)

        term_loss = (losses * b_d).sum() / (b_d.sum() + 1e-20)
        flow_loss = (losses * (1-b_d)).sum() / ((1-b_d).sum() + 1e-20)
        priority = _losses.detach().clone()

        if args.balanced_loss:
            loss = term_loss * args.leaf_coef + flow_loss
            priority[b_d == 1] *= args.leaf_coef
        else:
            loss = losses.mean()

        # ==========================================================================================
        # Long credit assignment
        # ASSUMPTION 1 batch has entire trajectories and they are ordered (s0,s1,s2,s0,s1,s2,s3...)
        # ASSUMPTION 2 consider trajectories of min 2 steps

        reinforce = torch.autograd.grad(loss, inflow, retain_graph=True)[0].detach()

        # torch.exp(qsa_p[parent_idx[i]]) - is Q(s,a) for state state[traj_steps[i]] from traj_id[i]
        parent_idx, traj_steps, traj_ids = [], [], []

        parent_mols, s_mols = mols
        parent_cnt, prev_done, traj_cnt = 0, 1, -1
        for ix, (xmol, pmols) in enumerate(zip(s_mols, parent_mols)):
            if hasattr(xmol, "_backward_parent_idx"):
                # Traj was constructed backward
                p_idx = xmol._backward_parent_idx
            elif hasattr(xmol, "_parent_deleted_block"):
                # Should be the one that was last (traj constructed by adding new block at end)
                p_idx = xmol._parent_deleted_block.index(len(xmol.blockidxs) - 1)
            else:
                # No parents method was called on the molecule (Action terminate
                p_idx = 0
                assert len(pmols) == 1, "what? no parent molecules"

            parent_idx.append(parent_cnt + p_idx)
            parent_cnt += len(pmols)
            if prev_done:
                traj_cnt += 1
                traj_steps.append(0)
                traj_ids.append(traj_cnt)
            else:
                traj_steps.append(traj_steps[-1] + 1)
                traj_ids.append(traj_cnt)

            prev_done = b_d[ix]

        parent_idx = torch.tensor(parent_idx, device=device, dtype=torch.long)
        traj_steps = torch.tensor(traj_steps, device=device, dtype=torch.long)
        traj_ids = torch.tensor(traj_ids, device=device, dtype=torch.long)

        # traj_qsa = torch.exp(qsa_p[parent_idx]) ?!?! or
        traj_qsa = qsa_p[parent_idx]

        t_corrections = []
        traj_loss = torch.zeros(traj_cnt + 1).to(device).to(self.float_ttype)

        for traj_end_step in range(1, traj_steps.max() + 1)[::-1]:
            tend_select = torch.where(traj_steps == traj_end_step)[0]
            inflow_reinf = reinforce[tend_select]

            t_correction = 0

            for minus_t in range(1, traj_end_step + 1):
                t_correction += traj_qsa[tend_select - 1] * inflow_reinf
            t_corrections.append(t_correction)
            traj_loss[traj_ids[tend_select]] += t_correction

        if t_avg_by_trajectory:
            t_correction_loss = (traj_loss / (traj_steps[torch.where(b_d == 1)] + 1)).mean()
        else:
            t_correction_loss = torch.mean(torch.tensor([torch.mean(x) for x in t_corrections]))

        loss += t_correction_loss * t_correction_coef
        # print(t_correction_loss.item())

        # ==========================================================================================

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
                exp_inflow.data.cpu().numpy(),
                exp_outflow.data.cpu().numpy(),
                b_r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                torch.autograd.grad(loss, qsa_p, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_s, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_p, retain_graph=True)[0].data.cpu().numpy(),
                self.train_num_mol / (time.time() - self._train_start_time),
                t_correction_loss.data.cpu().numpy(),
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

        return {"losses": priority.detach()}, log_metrics
