import cv2
from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from argparse import Namespace
import os
import warnings
from torch.multiprocessing import Process, Queue

import wandb

warnings.filterwarnings('ignore')
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
import torch
from torch.distributions.categorical import Categorical
from typing import List

from LambdaZero.utils import get_external_dirs

from LambdaZero.examples.lightrl.utils.utils import add_to_cfg, set_seed, setup_loggers

from LambdaZero.examples.generative_models.mols.mol_mdp_ext import BlockMoleculeDataExtended

from LambdaZero.examples.gflow.gflow_models import get_gflow_model
from LambdaZero.examples.gflow.utils import calc_metrics
from LambdaZero.examples.gflow.trainer import get_gflow_trainer
from LambdaZero.examples.gflow.proxy_wrappers import CandidateWrapper
from LambdaZero.examples.lightrl.utils.utils import LogTopStats, SummaryStats
from LambdaZero.examples.gflow.datasets import get_gflow_dataset

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

tmp_dir = os.environ['SLURM_TMPDIR'] + '/lztmp/'
os.makedirs(tmp_dir, exist_ok=True)


def log_stats_remote(conn_recv, conn_send, log_topk, do_dockscore):
    if do_dockscore:
        score_keys = ("proxy", "qed", "synth", "dockscore", "dscore")
    else:
        score_keys = ("proxy", "qed", "synth")

    log_stats = LogTopStats(topk=log_topk, unique_key="smiles", score_keys=score_keys)

    while True:
        cmd, recv = conn_recv.get()
        if cmd == 0:
            log_stats.collect(*recv)
        elif cmd == 1:
            stats = log_stats.log()
            stats["total_num_steps"] = recv
            conn_send.put(stats)


class LogTopKproc:
    def __init__(self, args: Namespace):
        self.log_stats_remote_send = log_stats_remote_send = Queue()
        self.log_stats_remote_recv = log_stats_remote_recv = Queue()
        log_proc_stats = Process(
            target=log_stats_remote,
            args=(log_stats_remote_send, log_stats_remote_recv, args.main.log_topk, args.main.log_dockscore)
        )
        log_proc_stats.start()
        self.recv_logtop = 0
        self.sent_logtop = 0
        self.log_topk_freq = args.main.log_topk_freq
        self._last_topk_step = 0
        self._no_score_infos = 0
        self._none_score_infos = 0
        self._collected = 0
        self._req_stats = {}

    def collect(self, infos: List[dict], total_num_steps: int):
        """
            Send infos for good candidates to keep buffer for topK
            Dict should have following keys with values ("proxy", "qed", "synth", "smiles", "score")
            TopK will be chosen based on the score value

            total_num_steps: number of steps at collection time
        """

        # If proxy exist -> rest of the scores exist
        send_info = []
        for sinfo in infos:
            self._collected += 1
            if "score" not in sinfo:
                self._no_score_infos += 1
            elif sinfo["score"] is None:
                self._none_score_infos += 1
            else:
                send_info.append(sinfo)

        if len(send_info) > 0:
            self.log_stats_remote_send.put((0, (infos,)))

        if total_num_steps > self._last_topk_step + self.log_topk_freq:
            self.send_stats_request(total_num_steps)
            self._last_topk_step = total_num_steps

    def get_stats(self):
        """ Non-blocking get """
        log_stats_remote_recv = self.log_stats_remote_recv
        ret_log_stats = []
        while not log_stats_remote_recv.empty():
            log_stats = log_stats_remote_recv.get()
            log_stats.update(self._req_stats[log_stats["total_num_steps"]])
            ret_log_stats.append(log_stats)
            self.recv_logtop += 1

        return ret_log_stats

    def _local_stats(self):
        collected = self._collected
        stats = {
            "topk_received": collected,
        }
        if self._collected > 0:
            stats["topk_no_score"] = self._no_score_infos / collected
            stats["topk_none_score"] = self._none_score_infos / collected
            stats["topk_has_score"] = (self._no_score_infos + self._none_score_infos) / collected

        self._collected = 0
        self._no_score_infos = 0
        self._none_score_infos = 0
        return stats

    def send_stats_request(self, total_num_steps: int):
        """
            Non blocking send request. LogTopStats will start calculating TopK with current buffer
        """
        if (self.sent_logtop - self.recv_logtop) < 2:
            self._req_stats[total_num_steps] = self._local_stats()
            self.log_stats_remote_send.put((1, total_num_steps))
            self.sent_logtop += 1
        else:
            print(f"NOT GOOD. Skipping log top {total_num_steps}")


def sample_and_update_dataset(args, model, generator_dataset, proxy):
    # generator_dataset.set_sampling_model(model, docker, sample_prob=args.sample_prob)
    # sampler = generator_dataset.start_samplers(8, args.num_samples)
    print("Sampling")
    # sampled_mols = sampler()
    # generator_dataset.stop_samplers_and_join()
    # import pdb; pdb.set_trace()
    mdp = generator_dataset.mdp
    nblocks = mdp.num_blocks
    sampled_mols = []
    sampled_infos = []
    rews = []
    smis = []
    max_trials = args.num_samples
    while len(sampled_mols) < args.num_samples:
        mol = BlockMoleculeDataExtended()
        for i in range(args.max_blocks):
            s = mdp.mols2batch([mdp.mol2repr(mol)])
            stem_o, mol_o = model(s)
            logits = torch.cat([stem_o.flatten(), mol_o.flatten()])
            if i < args.min_blocks:
                logits[-1] = -1000
            cat = Categorical(logits=logits)
            act = cat.sample().item()
            if act == logits.shape[0] - 1:
                break
            else:
                act = (act % nblocks, act // nblocks)
                mol = mdp.add_block_to(mol, block_idx=act[0], stem_idx=act[1])
            if not len(mol.stems):
                break
        if mol.mol is None:
            print('skip', mol.blockidxs, mol.jbonds)

            if max_trials > 0:
                max_trials -= 1
                continue
            else:
                print("Reached Max trails!!!")
                break

        # print('here')
        res_scores, infos = proxy([mol])
        # This is actually proxy score ( Could be default min if not candidate)
        mol.reward = res_scores[0]
        smis.append(mol.smiles)
        rews.append(mol.reward)
        sampled_mols.append(mol)
        sampled_infos.append(infos[0])
        # print(infos)
        
    print("Computing distances")
    dists = []
    for m1, m2 in zip(sampled_mols, sampled_mols[1:] + sampled_mols[:1]):
        dist = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m1.mol),
                                                 Chem.RDKFingerprint(m2.mol))
        dists.append(dist)
    print("Get batch rewards")
    rewards = []
    for m in sampled_mols:
        rewards.append(m.reward)

    candidate_proxy = [x["proxy"] for x in sampled_infos if x["proxy"] is not None]
    return rews, smis, \
           {'dists': dists, 'proxy': rewards}, \
           {**calc_metrics("eval_proxy", candidate_proxy),
            f'eval_dists_mean': np.mean(dists), f'eval_dists_sum': np.sum(dists)},\
           sampled_infos


def preprocess_args(args: Namespace):
    args.cuda = args.main.use_gpu and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if args.main.seed == 0:
        args.main.seed = args.run_id + 1

    print("SLURM_ARRAY_TASK_ID", os.environ.get('SLURM_ARRAY_TASK_ID', None))
    print("SLURM_ARRAY_JOB_ID", os.environ.get('SLURM_ARRAY_JOB_ID', None))
    print(f"SEED: {args.main.seed}")

    # -- Set seed
    add_to_cfg(args, None, "device", args.device)
    add_to_cfg(args, None, "seed", args.main.seed)
    args.seed = args.main.seed
    set_seed(args.seed, args.cuda, getattr(args, "cuda_deterministic", False))

    # -- Init wandb logger
    args.use_wandb = args.main.plot
    out_dir = setup_loggers(args)
    add_to_cfg(args, None, "out_dir", out_dir)

    return args


def run(args):
    # -- Some initial configurations (Wandb logger, seed, device ...)
    args = preprocess_args(args)
    device = args.device
    do_plot = args.main.plot
    out_dir = args.out_dir

    # -- Load local variables
    rews, smis = [], []
    main_args = args.main
    gflow_args = args.gflow
    gflow_dataset_args = args.gflow_dataset
    gflow_model_args = args.gflow_model
    num_outer_loop_iters = main_args.num_outer_loop_iters

    # -- Load classes to generate scores
    proxy = CandidateWrapper(args.proxy)
    # proxy = load_proxy(args.proxy)
    # proxy.to(device)

    # ==============================================================================================
    # -- GFlow training dataset (Generates samples and manages training batches)
    _gen_class = get_gflow_dataset(gflow_dataset_args)

    if getattr(gflow_dataset_args, "wrapper", None) is not None:
        gflow_dataset_args.name = gflow_dataset_args.wrapper
        _wrapped_class = get_gflow_dataset(gflow_dataset_args)
        _wrapped_class.__bases__ = (_gen_class,)
        _gen_class = _wrapped_class

    gen_model_dataset = _gen_class(gflow_dataset_args, args.device)

    # ==============================================================================================
    # -- Load model for GFlow
    model = get_gflow_model(gflow_model_args, mdp=gen_model_dataset.mdp)
    model = model.double() if gflow_model_args.floatX == 'float64' else model
    model.to(device)

    # ==============================================================================================
    # -- Data feeder
    gen_model_dataset.set_sampling_model(model, proxy, sample_prob=gflow_dataset_args.sample_prob)

    # -- Evaluation args
    gflow_eval_args = args.gflow_eval
    eval_freq = gflow_eval_args.eval_freq
    last_eval_step = 0

    # RUN TOPK
    # Get remote calculation of logs for topk stats (because we also calculate dockscore)
    if args.main.log_topk_freq > 0:
        log_top_k = LogTopKproc(args)

    # This logs statistics for MIN and MAX values for everything plotted (summary for wandb)
    log_summary = SummaryStats(log_wandb=do_plot)

    plot_train_k = ["train_loss_mean", "train_num_mol", "total_num_steps",
                    "train_num_trajectories", "train_r_mean", "train_r_max", "FPS_mean"]
    plot_eval_k = ["eval_proxy_mean", "eval_proxy_cnt"]
    plot_top_k = ["top100_proxy_mean", "top100_proxy_count", "top100_count"]

    def plot_keys(keys: List[str], metrics: dict):
        print_txt = " | ".join([f"{k}: {metrics[k]:.2f}" for k in keys if k in metrics])
        return print_txt

    # ==============================================================================================
    # -- Load GFLOW trainer
    if gflow_args.name == "TrainGFlow":
        assert False, "Changed dataloader. Need GenModelDataset for this trainer"

    gflow_trainer = get_gflow_trainer(
        gflow_args, model=model, proxy=proxy, dataset=gen_model_dataset, do_save=False
    )
    num_iterations = gflow_args.num_iterations

    for i in range(num_outer_loop_iters):
        print(f"Starting step: {i}")
        # Initialize model and dataset for training generator
        # train model with with proxy

        for epoch in range(num_iterations):
            model, _, training_metrics = gflow_trainer.train_epoch(epoch)
            train_data_stats, train_mol_infos = gen_model_dataset.get_stats()
            training_metrics.update(train_data_stats)
            num_steps = training_metrics["total_num_steps"]

            print(f"[E {epoch}] [TRAIN] {plot_keys(plot_train_k, training_metrics)}")
            training_metrics["epoch"] = epoch
            training_metrics["outer_epoch"] = i

            if eval_freq > 0 and num_steps > last_eval_step + eval_freq:
                r, s, batch_metrics, log_metrics, sample_infos = sample_and_update_dataset(
                    gflow_eval_args, model, gen_model_dataset, proxy
                )
                train_mol_infos += sample_infos
                log_summary.update(log_metrics)
                training_metrics.update(log_metrics)
                last_eval_step = num_steps
                print("-"*100, f"\n[E {epoch}] [EVAL] {plot_keys(plot_eval_k, log_metrics)}\n", "-"*100)

            if do_plot:
                wandb.log(training_metrics)

            if args.main.log_topk_freq > 0:
                # It starts calculating topk Automatically every args.main.log_topk_freq steps
                log_top_k.collect(train_mol_infos, num_steps)

                # Get async TopK logs (Has a different total_num_steps than previous logs)
                all_stats_top_k = log_top_k.get_stats()

                for stats_top_k in all_stats_top_k:
                    print("-"*100, f"\n[E {epoch}] [TOPK] {plot_keys(plot_top_k, stats_top_k)}\n", "-"*100)
                    log_summary.update(stats_top_k)
                    if do_plot:
                        wandb.log(stats_top_k)

        gflow_trainer.stop_everything()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except:
        print("already set context")

    from LambdaZero.examples.lightrl.utils.utils import parse_opts
    run(parse_opts())
