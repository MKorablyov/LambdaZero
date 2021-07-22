import time
import torch
from copy import deepcopy
import os
import pickle
import gzip
import pdb
import numpy as np
from argparse import Namespace

from LambdaZero.examples.gflow.main_flow import Dataset as GenModelDataset
from LambdaZero.examples.gflow.utils import LogMetrics, TimeStats


_stop = [None]


class TrainGFlow:
    def __init__(self,
                 args,
                 model=None, proxy=None, dataset: GenModelDataset=None, do_save=True):

        self.args = args
        self.debug_no_threads = debug_no_threads = getattr(args, "debug_no_threads", False)
        self.device = device = args.device
        self._log_min_new = getattr(args, "log_min_new", 64)
        self.do_save = do_save

        self.model = model
        self.dataset = dataset
        self.exp_dir = args.out_dir
        mbsize = args.mbsize

        # -- Model & optimizer
        model = model.double()
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          args.learning_rate, weight_decay=args.weight_decay,
                                          betas=(args.opt_beta, args.opt_beta2),
                                          eps=args.opt_epsilon)

        if args.bootstrap_tau > 0:
            self.target_model = deepcopy(model)

        # -- Dataset start
        dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

        if not debug_no_threads:
            self.sampler = dataset.start_samplers(8, mbsize)

        def stop_everything():
            print('What joining')
            dataset.stop_samplers_and_join()

        self.stop_everything = stop_everything

        _stop[0] = stop_everything

        # -- Log stuff
        self._train_losses_k = [
            "train_loss", "train_term_loss", "train_flow_loss", "train_term_loss",
            "train_flow_loss"
        ]
        self._train_infos_k = [
            "tinfo_term_loss", "tinfo_flow_loss", "exp_inflow", "exp_outflow",
            "train_r", "train_mols", "train_params", "grad_qsa_p",
            "grad_stem_out_s", "grad_stem_out_p"
        ]
        self._log_metrics = self._train_losses_k + ["train_r", "train_sampled_dockscore"]
        self._train_metrics = LogMetrics(self._train_losses_k + self._train_infos_k +
                                         ["train_sampled_dockscore"])

        if do_save:
            exp_dir = f'{args.out_dir}/{args.array}_{args.run}/'
            os.makedirs(exp_dir, exist_ok=True)

        self.train_num_trajectories = 0
        self.train_num_mol = 0
        self.time_last_check = time.time()
        self._last_ep_sampled_mol = 0
        self._ts_stats = Namespace(
            sample=TimeStats(),
            train=TimeStats(),
            inflow=TimeStats(),
            outflow=TimeStats(),
            lossbackward=TimeStats(),
            optstep=TimeStats(),
        )

    def save_stuff(self):
        model = self.model
        exp_dir = self.exp_dir
        dataset = self.dataset
        metrics = self._train_metrics

        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols, gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'wb'))

        _train_losses = list(zip(metrics.get_values(self._train_losses_k, dict_format=False)))
        _train_infos = list(zip(metrics.get_values(self._train_infos_k, dict_format=False)))

        pickle.dump({
            'time_start': self.time_start,
            'time_now': time.time(),
            'args': self.args,
            "train_infos": _train_infos,
            "train_losses": _train_losses},
            gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        pickle.dump(self.args.train_infos, gzip.open(f'{exp_dir}/train_info.pkl.gz', 'wb'))

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

        _t_losses_k, _t_infos_k = self._train_losses_k, self._train_infos_k

        tf = lambda x: torch.tensor(x, device=device).to(torch.float64 if args.floatX == 'float64' else torch.float32)
        clip_loss = tf([args.clip_loss])

        self._ts_stats.sample.start()

        if not debug_no_threads:
            r = self.sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    self.stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            p, pb, a, r, s, d, mols = r
        else:
            p, pb, a, r, s, d, mols = dataset.sample2batch(dataset.sample(args.mbsize))
        self._ts_stats.sample.end()

        self._ts_stats.train.start()
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]

        self.train_num_trajectories += args.mbsize
        self.train_num_mol += ntransitions

        self._ts_stats.inflow.start()
        # state outputs
        if args.bootstrap_tau > 0:
            with torch.no_grad():
                stem_out_s, mol_out_s = self.target_model(s, None)
        else:
            stem_out_s, mol_out_s = model(s, None)
        # parents of the state outputs
        stem_out_p, mol_out_p = model(p, None)
        # index parents by their corresponding actions
        qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
        # then sum the parents' contribution, this is the inflow
        exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                      .index_add_(0, pb, torch.exp(qsa_p))) # pb is the parents' batch index
        inflow = torch.log(exp_inflow + args.log_reg_c)
        self._ts_stats.inflow.end()

        self._ts_stats.outflow.start()
        # sum the state's Q(s,a), this is the outflow
        exp_outflow = model.sum_output(s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(args.log_reg_c + r + exp_outflow * (1-d))
        if args.do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) / (s.nblocks * args.max_blocks)).pow(2)
        else:
            losses = _losses = (inflow - outflow_plus_r).pow(2)
        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)
        self._ts_stats.outflow.end()

        self._ts_stats.lossbackward.start()
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
                exp_inflow.data.cpu().numpy(),
                exp_outflow.data.cpu().numpy(),
                r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                torch.autograd.grad(loss, qsa_p, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_s, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_p, retain_graph=True)[0].data.cpu().numpy(),
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

            if not epoch % 1000 and do_save:
                self.save_stuff()

        # New molecules
        new_molecules = dataset.sampled_mols[self._last_ep_sampled_mol:]
        if len(new_molecules) > self._log_min_new:
            self._last_ep_sampled_mol = len(dataset.sampled_mols)
            new_mol_dockscore = [dataset.inv_r2r(x[0]) for x in new_molecules]
            t_metrics.update(["train_sampled_dockscore"], [new_mol_dockscore])

        log_metrics = t_metrics.get_stats(self._log_metrics)
        log_metrics.update({
            "train_num_sampled_mols": len(dataset.sampled_mols),
            "train_num_mol": self.train_num_mol,
            "total_num_steps": self.train_num_mol,
            "train_num_trajectories": self.train_num_trajectories
        })

        # self.stop_everything() # ONLY when end
        if do_save:
            self.save_stuff()

        for k, v in self._ts_stats.__dict__.items():
            print(f"{k}: {v.stats():.6f}")

        return model, dataset, log_metrics
