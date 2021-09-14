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


class TrainGFlowFwdBackLeafV(BasicTrainer):
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
        log_log_reg_c = torch.log(tf([args.log_reg_c])[0])

        b_p, b_pb, b_a, b_r, b_s, b_d, mols = train_batch

        ntransitions = b_r.shape[0]

        self.train_num_mol += ntransitions

        # Calculate forward / backward policy log probs and values
        p_stem_out, p_mol_out, _ = model.run_model(b_p)
        s_stem_out, s_mol_out, s_jbond_out = model.run_model(b_s)
        s_mol_out, s_v_out, s_v_term = s_mol_out[:, :1], s_mol_out[:, 1], s_mol_out[:, 2]
        p_mol_out, p_v_out, p_v_term = p_mol_out[:, :1], p_mol_out[:, 1], p_mol_out[:, 2]  # Assume two values out / molecule

        # print("P", p_v_out.std().item(), "S", s_v_out.std().item())
        split_v = False
        non_term = b_a[:, 0] > -1
        term = ~non_term
        if split_v:
            # print("WTFFFFF")
            # Terminal ones have done True so don't need to correct
            s_v_out = torch.log(torch.exp(s_v_out) + torch.exp(s_v_term.detach()))

            # We should correct based on True terminal value
            term_p_v = b_r.clone()
            term_p_v[non_term] = torch.exp(p_v_term[non_term].detach())
            p_v_out = torch.log(torch.exp(p_v_out) + term_p_v)

        if clip_policy > 0:
            for ix, p_out in enumerate([p_stem_out, p_mol_out, s_stem_out, s_mol_out, s_jbond_out]):
                p_out.clip_(-clip_policy, clip_policy)

        # Forward Q approximation based on V(p) * pi_fwd(a, p)
        # f_action_logprob = -model.action_negloglikelihood(p, a, 0, p_stem_out, p_mol_out)
        # fwd_flow = torch.log(args.log_reg_c + torch.exp(p_v_out + f_action_logprob))
        #
        f_action_logprob = -model.action_negloglikelihood(b_p, b_a, 0, p_stem_out, p_mol_out)
        fwd_flow = torch.logaddexp(log_log_reg_c, p_v_out + f_action_logprob)

        do_log_terminal = False
        if do_log_terminal:
            # index parents by their corresponding actions
            qsa_p = model.index_output_by_action(b_p, p_stem_out, p_mol_out[:, 0], b_a)

            # then sum the parents' contribution, this is the inflow
            exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                          .index_add_(0, b_pb, torch.exp(qsa_p)))  # pb is the parents' batch index
            inflow = torch.log(exp_inflow + args.log_reg_c)

        # Calculate backward policy log prob - based on delete per bond atom prediction
        # bond_act = torch.zeros_like(b_a)
        # b_action_logprob = -model.action_negloglikelihood_bonds(s, bond_act, s_jbond_out)

        b_cnt = torch.tensor(b_s.__slices__['bonds'])
        pjbonds = (b_cnt[1:] - b_cnt[:-1]).clamp(min=1)
        # b_action_logprob = torch.log(1 / pjbonds).to(p_stem_out.dtype).to(p_stem_out.device)

        # Backward Q approximation based on V(s) * pi_fwd(a, s)
        # bck_flow = torch.exp(s_v_out + b_action_logprob)
        bck_flow = torch.exp(s_v_out) / pjbonds.to(p_stem_out.device)
        # bck_flow = torch.exp(s_v_out)  # TODO

        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        # bck_flow_plus_r = torch.log(args.log_reg_c + b_r + bck_flow * (1-b_d))
        bck_flow_plus_r = torch.log(args.log_reg_c + b_r + bck_flow * (1-b_d))

        # print(f_action_logprob.size(), fwd_flow.size(), bck_flow.size(), b_r.size(), bck_flow_plus_r.size())

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

        if split_v and term.sum() > 0:
            # calculate loss for term value
            loss_term_act = (b_r[term] - torch.exp(p_v_term)[term]).pow(2).mean()
            loss += loss_term_act

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

        if False and not epoch % 50:
            loss_sort_idx = _losses.argsort().flip((0,))
            print(f"_losses {_losses[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"b_action_logprob {b_action_logprob[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"f_action_logprob {f_action_logprob[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"Rew {b_r[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"Done {b_d[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"eS_v_out {torch.exp(s_v_out)[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"eP_v_out {torch.exp(p_v_out)[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"efwd_flow {torch.exp(fwd_flow)[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"ebck_flow_plus_r {torch.exp(bck_flow_plus_r)[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"bck_flow {bck_flow[loss_sort_idx].data.cpu().numpy().tolist()}")
            print(f"act {b_a[loss_sort_idx].data.cpu().numpy().tolist()}")

            pmol = [mols[0][imol][0] for imol in loss_sort_idx]
            smol = [mols[1][imol] for imol in loss_sort_idx]
            print(f"pmol_size {[len(x.blockidxs) for x in pmol]}")
            print(f"pmol_bs {[x.blockidxs for x in pmol]}")
            print(f"pmol_jbonds {[x.jbonds for x in pmol]}")
            # if epoch > 500:
            #     import pdb;
            #     pdb.set_trace()

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

