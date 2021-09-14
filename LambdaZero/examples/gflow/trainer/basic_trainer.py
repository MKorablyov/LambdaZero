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


class BasicTrainer:
    def __init__(self,
                 args,
                 model=None, proxy=None, dataset: DataGenerator=None, do_save=True,
                 extra_t_info=[]):

        self.args = args
        self.debug_no_threads = debug_no_threads = getattr(args, "debug_no_threads", False)
        self.device = device = args.device
        self.do_save = do_save

        self.model = model
        self.dataset = dataset
        self.exp_dir = args.out_dir

        # Determine if to train on fixed number of trajectories or transitions
        self._batch_size_in_traj, batch_size = True, getattr(args, "batch_size", 0)
        if batch_size is not None and batch_size != 0:
            self._batch_size_in_traj, self._batch_size = False, batch_size
        else:
            self._batch_size = getattr(args, "traj_batch_size", 0)

        self.float_ttype = torch.float64 if args.floatX == 'float64' else torch.float32

        # -- Model & optimizer
        if args.floatX == 'float64':
            model = model.double()

        _optim = getattr(torch.optim, args.optim)
        self.optimizer = _optim(model.parameters(), **vars(args.optim_args))

        if args.bootstrap_tau > 0:
            self.target_model = deepcopy(model)

        # -- Log stuff
        self._train_losses_k = [
            "train_loss", "train_term_loss", "train_flow_loss", "train_term_loss",
            "train_flow_loss"
        ]
        self._train_infos_k = [
            "tinfo_term_loss", "tinfo_flow_loss", "exp_inflow", "exp_outflow",
            "train_r", "train_mols", "train_params", "grad_qsa_p",
            "grad_stem_out_s", "grad_stem_out_p", "FPS"
        ] + extra_t_info

        # This will be the metrics that actually get logged to plots
        self._log_metrics = self._train_losses_k + extra_t_info + ["train_r", "FPS"]
        self._train_metrics = LogMetrics(self._train_losses_k + self._train_infos_k)

        if do_save:
            exp_dir = f'{args.out_dir}/{args.array}_{args.run}/'
            os.makedirs(exp_dir, exist_ok=True)

        self.train_num_mol = 0
        self.time_last_check = time.time()
        self._train_start_time = None

        # We can use this to time stuff :D
        self._ts_stats = Namespace(
            sample=TimeStats(),
            train=TimeStats(),
            inflow=TimeStats(),
            outflow=TimeStats(),
            lossbackward=TimeStats(),
            optstep=TimeStats(),
        )

    def train_batch_info(self):
        return self._batch_size_in_traj, self._batch_size

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

        ntransitions = b_r.shape[0]

        self.train_num_mol += ntransitions

        # state outputs
        if args.bootstrap_tau > 0:
            with torch.no_grad():
                stem_out_s, mol_out_s = self.target_model(b_s)
        else:
            stem_out_s, mol_out_s = model(b_s)

        # parents of the state outputs
        stem_out_p, mol_out_p = model(b_p)

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


        if args.do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) / (b_s.nblocks * args.max_blocks)).pow(2)
        else:
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
            """ 
                add some metrics to log (just _log_metrics of them will be logged to wandb though) 
            """
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

        return {"losses": priority.detach()}, log_metrics

    def stop_everything(self):
        pass

    def compute_loss(self, train_batch: TrainBatch):
        """ We use it for example for priority sampling - don't train just get loss """
        model = self.model
        device = self.device
        dataset = self.dataset
        args = self.args
        tf = lambda x: torch.tensor(x, device=device).to(self.float_ttype)
        clip_loss = tf([args.clip_loss])

        b_p, b_pb, b_a, b_r, b_s, b_d, mols = train_batch
        ntransitions = b_r.shape[0]

        stem_out_s, mol_out_s = model(b_s)
        stem_out_p, mol_out_p = model(b_p)

        qsa_p = model.index_output_by_action(b_p, stem_out_p, mol_out_p[:, 0], b_a)

        exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                      .index_add_(0, b_pb, torch.exp(qsa_p)))  # pb is the parents' batch index
        inflow = torch.log(exp_inflow + args.log_reg_c)
        exp_outflow = model.sum_output(b_s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        outflow_plus_r = torch.log(args.log_reg_c + b_r + exp_outflow * (1-b_d))

        if args.do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) / (b_s.nblocks * args.max_blocks)).pow(2)
        else:
            losses = _losses = (inflow - outflow_plus_r).pow(2)
        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)
        term_loss = (losses * b_d).sum() / (b_d.sum() + 1e-20)
        flow_loss = (losses * (1-b_d)).sum() / ((1-b_d).sum() + 1e-20)
        priority = _losses.detach().clone()
        if args.balanced_loss:
            priority[b_d == 1] *= args.leaf_coef
        return priority