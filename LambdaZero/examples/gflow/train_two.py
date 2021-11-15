import time

import cv2
import sys
import pandas as pd
import torch.multiprocessing as mp
from collections import Counter, deque
import sys
import os
import time

try:
    mp.set_start_method('spawn')
except:
    print("already set context", mp.get_context())

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
from scipy import stats

from LambdaZero.utils import get_external_dirs

from LambdaZero.examples.lightrl.utils.utils import add_to_cfg, set_seed, setup_loggers

from LambdaZero.examples.generative_models.mols.mol_mdp_ext import BlockMoleculeDataExtended

from LambdaZero.examples.gflow.gflow_models import get_gflow_model
from LambdaZero.examples.gflow.utils import calc_metrics
from LambdaZero.examples.gflow.trainer import get_gflow_trainer
from LambdaZero.examples.gflow.proxy_wrappers import PROXY_WRAPPERS
from LambdaZero.examples.lightrl.utils.utils import SummaryStats
from LambdaZero.examples.gflow.datasets import get_gflow_dataset
from LambdaZero.examples.gflow.utils import LogTopKproc, show_histogram

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

tmp_dir = os.environ['SLURM_TMPDIR'] + '/lztmp/'
os.makedirs(tmp_dir, exist_ok=True)


def analyse_eval(args, generator_dataset, proxy, model, epoch, sampled_infos,
                 smis, rews, blocksize, entropy, exp_sum):

    df = pd.DataFrame(np.array([blocksize, entropy, exp_sum]).transpose(),
                      columns=["blocksize", "entropy", "exp_sum"])
    print(df.groupby("blocksize")["entropy"].mean())
    print(df.groupby("blocksize")["exp_sum"].mean())
    candidate_proxy = [x["proxy"] for x in sampled_infos if x["proxy"] is not None]
    candidate_scores = [x["score"] for x in sampled_infos if x["score"] is not None]
    all_scores = [x["score"] if x["score"] is not None else proxy._default_score * -1 for x in sampled_infos]

    drcount = generator_dataset.debug_r_counter = \
        getattr(generator_dataset, "debug_r_counter", deque(maxlen=args.num_samples*10))
    dscount = generator_dataset.debug_s_counter = \
        getattr(generator_dataset, "debug_s_counter", deque(maxlen=args.num_samples*10))
    dssmis = generator_dataset.debug_s_smis = \
        getattr(generator_dataset, "debug_s_smis", deque(maxlen=args.num_samples*10))
    smpl_uniq = getattr(generator_dataset, "smpl_uniq", set())
    smpl_uniq.update(smis)
    dssmis += smis
    drcount += rews
    dscount += all_scores

    e_r_bin_cnt, e_r_bin_edge = np.histogram(drcount, bins=64)

    print(f"EVAL (num uniq {len(smpl_uniq)}) with rew freq: {list(zip(e_r_bin_cnt, e_r_bin_edge))}")
    print(f"EVAL (num uniq {len(smpl_uniq)}) with score freq: {list(zip(*np.histogram(dscount)))}")

    debug_logs = {}
    debug_hist = None

    # -- Debug code for debug environment
    if hasattr(proxy, "_tgt_score_edges"):
        e_bin_cnt, e_bin_edge = np.histogram(dscount, bins=proxy._tgt_score_edges)

        tgt_bins = proxy._tgt_r_bin_probs * len(dscount)

        print(f"[DEBUG] Eval bin cnt (in tgt edge): {e_bin_cnt.tolist()}")
        print(f"[DEBUG] EVAL target bin_cnt {np.around(tgt_bins, 1).tolist()}")
        print(f"[DEBUG] Eval diff to target: {np.around(e_bin_cnt - tgt_bins, 1).tolist()}")
        e_prob = e_bin_cnt / len(dscount)

        mae = np.abs(e_prob - proxy._tgt_r_bin_probs)
        debug_logs = {"evalH_tgt_mae": np.mean(mae), "evalH_tgt_max": np.max(mae),
                      "debug_found_top": proxy.found_top()}

        # plot histogram
        # Plot history of eval samples vs tgt
        if not hasattr(proxy, "_evalh_history"):
            proxy._evalh_history = [proxy._tgt_r_bin_probs * dscount.maxlen]
            proxy._evalh_err_history = [np.zeros_like(proxy._tgt_r_bin_probs)]

        evalh_history = proxy._evalh_history
        evalh_err_history = proxy._evalh_err_history

        evalh_history.append(e_bin_cnt)
        evalh_err_history.append(e_bin_cnt - tgt_bins)

        show_edges = proxy._show_tgt_score_edges[1:]
        zrange = proxy._tgt_probs_zrange

        debug_hist = [
            ("mevalH_debug_h", show_histogram(evalh_history, dscount.maxlen, show_edges, zrange=zrange)),
            ("mevalH_debug_err_h", show_histogram(evalh_err_history, dscount.maxlen, show_edges))
        ]

        # run debug
        if hasattr(proxy, "_debug_batch"):
            debug_batch = proxy._debug_batch
            out_dir = f"{args.out_dir}/debug"

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
                torch.save({"debug_batch": debug_batch}, f"{out_dir}/debug_batch.pk")

            with torch.no_grad():
                parents_out = model(debug_batch.parents.clone())
                state_out = model(debug_batch.state.clone())
            torch.save(
                {"parents_out": parents_out, "state_out": state_out, **debug_logs},
                f"{out_dir}/eval_{epoch}.pk"
            )

    print(f"PROXY (num uniq {len(proxy._seen)}) with score freq: "
          f"{list(zip(*np.histogram(proxy._seen_scores)))}")

    return {
        "drcount": drcount,
        "dscount": dscount,
        "candidate_scores": candidate_scores,
        "debug_hist": debug_hist,
        "candidate_proxy": candidate_proxy,
        "debug_logs": debug_logs,
    }


def sample_and_update_dataset_batch(args, model, generator_dataset, proxy, epoch):
    """ Sample from learned gflow + some debugging code """
    print("[Evaluation Sampling]")

    mdp = generator_dataset.mdp
    nblocks = mdp.num_blocks
    sampled_mols, sampled_infos, rews, smis = [], [], [], [],
    blocksize, entropy, exp_sum = [], [], []

    max_trials = args.num_samples
    batch_size = 128

    batch = [BlockMoleculeDataExtended() for _ in range(batch_size)]

    while len(sampled_mols) < args.num_samples and max_trials > 0:
        states = mdp.mols2batch([mdp.mol2repr(mol) for mol in batch])
        with torch.no_grad():
            stem_o, mol_o = model(states)

        stem_idx = states.__slices__["stems"]

        for imol in range(len(batch)):
            done = False
            s_o, m_o = stem_o[stem_idx[imol]: stem_idx[imol + 1]], mol_o[imol: imol + 1]

            logits = torch.cat([s_o.flatten(), m_o.flatten()])
            if len(batch[imol].blockidxs) < args.min_blocks:
                logits[-1] = -1000  # TODO hardcoded - value is used in other places

            cat = Categorical(logits=logits)
            blocksize.append(len(batch[imol].blockidxs))
            entropy.append(cat.entropy().item())
            exp_sum.append(torch.exp(logits).sum().item())

            act = cat.sample().item()
            if act == logits.shape[0] - 1:
                done = True
            else:
                act = (act % nblocks, act // nblocks)
                batch[imol] = mdp.add_block_to(batch[imol], block_idx=act[0], stem_idx=act[1])

            if not len(batch[imol].stems):
                done = True

            if len(batch[imol].blockidxs) >= args.max_blocks or done:
                mol = batch[imol]
                batch[imol] = BlockMoleculeDataExtended()
                if mol.mol is None:
                    print('skip', mol.blockidxs, mol.jbonds)
                    max_trials -= 1
                    continue

                r, info = generator_dataset._get_reward(mol)

                # add to online mols from generator so they can be used in training
                generator_dataset._add_mol_to_online(r, r, mol)

                # This is actually proxy score ( Could be default min if not using candidate score)
                mol.reward = r
                smis.append(mol.smiles)
                rews.append(mol.reward)
                sampled_mols.append(mol)
                sampled_infos.append(info)

    # ==============================================================================================
    # -- DEBUG CODE  # TODO messy / refactor or move somewhere else
    debug_info = analyse_eval(
        args=args, generator_dataset=generator_dataset, proxy=proxy, model=model, epoch=epoch,
        sampled_infos=sampled_infos, smis=smis, rews=rews, blocksize=blocksize, entropy=entropy,
        exp_sum=exp_sum)

    drcount, dscount, candidate_scores, debug_hist, candidate_proxy, debug_logs = \
        debug_info["drcount"], debug_info["dscount"], \
        debug_info["candidate_scores"], debug_info["debug_hist"], debug_info["candidate_proxy"], \
        debug_info["debug_logs"]

    # ==============================================================================================

    eval_r_90 = np.mean(drcount)
    eval_score_90 = np.mean(dscount)
    print(f"EVAL R 90%: {eval_r_90}")
    print(f"EVAL Score 90%: {eval_score_90}")

    print("Computing distances")
    dists = []
    for m1, m2 in zip(sampled_mols, sampled_mols[1:] + sampled_mols[:1]):
        dist = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m1.mol),
                                                 Chem.RDKFingerprint(m2.mol))
        dists.append(dist)

    return candidate_scores, smis, \
           {'dists': dists, "rews": rews, "debug_hist": debug_hist}, \
           {**calc_metrics("eval_proxy", candidate_proxy), **debug_logs,
            f'eval_dists_mean': np.mean(dists), f'eval_dists_sum': np.sum(dists),
            f"eval_r_90": eval_r_90, f"eval_s_90": eval_score_90,
            "eval_num_blocks_mean": np.mean([len(x.blockidxs) for x in sampled_mols]),
            # f"evalH_tgt_std": np.std(e_r_bin_cnt - tgt_bin),
            f"proxy_seen_count": len(proxy._seen),
            # f"proxy_seen_sum_r": sum_r_seen,
            "proxy_seen_nan_count": proxy._seen_nan_count},\
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

    # -- For grid experiment change Ref everywhere manually
    add_to_cfg(args, None, "repr_type", args.gflow_dataset.mdp_init.repr_type)
    add_to_cfg(args, None, "floatX", args.gflow_dataset.mdp_init.floatX)
    add_to_cfg(args, None, "min_blocks", args.gflow_dataset.min_blocks)
    add_to_cfg(args, None, "max_blocks", args.gflow_dataset.max_blocks)
    args.gflow_eval.eval_freq = args.main.eval_freq

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
    proxy = PROXY_WRAPPERS[args.proxy.name](args.proxy)
    proxy.to(device)

    # -- Load classes to generate scores
    # TODO D
    args.proxy_two.device = device
    args.proxy_two.floatX = gflow_model_args.floatX
    proxy_two = PROXY_WRAPPERS[args.proxy_two.name](args.proxy_two)
    proxy_two.to(device)

    # ==============================================================================================
    # -- GFlow training dataset (Generates samples and manages training batches)
    _gen_class = get_gflow_dataset(gflow_dataset_args)
    _wrapper = getattr(gflow_dataset_args, "wrapper", None)
    if _wrapper is not None and _wrapper != "":
        gflow_dataset_args.name = gflow_dataset_args.wrapper
        _wrapped_class = get_gflow_dataset(gflow_dataset_args)
        _wrapped_class.__bases__ = (_gen_class,)
        _gen_class = _wrapped_class

    gen_model_dataset = _gen_class(gflow_dataset_args, args.device)

    # ==============================================================================================
    # -- GFlow TWO (e.g. error) training dataset (Generates samples and manages training batches)
    # TODO D
    gflow_dataset_args_two = args.gflow_dataset_two  # TODO D 2 copy in config

    _gen_class = get_gflow_dataset(gflow_dataset_args_two)
    _wrapper = getattr(gflow_dataset_args_two, "wrapper", None)
    if _wrapper is not None and _wrapper != "":
        gflow_dataset_args_two.name = gflow_dataset_args_two.wrapper
        _wrapped_class = get_gflow_dataset(gflow_dataset_args_two)
        _wrapped_class.__bases__ = (_gen_class,)
        _gen_class = _wrapped_class

    gen_model_dataset_two = _gen_class(gflow_dataset_args_two, args.device)

    # ==============================================================================================
    # -- Load model for GFlow
    model = get_gflow_model(gflow_model_args, mdp=gen_model_dataset.mdp)
    model = model.double() if gflow_model_args.floatX == 'float64' else model
    model.to(device)

    # ==============================================================================================
    # -- Load model for GFlow TWO
    model_two = get_gflow_model(gflow_dataset_args_two, mdp=gen_model_dataset_two.mdp)
    model_two = model_two.double() if gflow_dataset_args_two.floatX == 'float64' else model_two
    model_two.to(device)

    # ==============================================================================================
    # -- Data feeder
    gen_model_dataset.set_sampling_model(model, proxy, sample_prob=gflow_dataset_args.sample_prob)
    gen_model_dataset_two.set_sampling_model(model_two, proxy_two, sample_prob=gen_model_dataset_two.sample_prob)

    # -- Evaluation args
    gflow_eval_args = args.gflow_eval
    eval_freq = gflow_eval_args.eval_freq
    last_eval_step = 0
    if hasattr(proxy, "debug"):
        proxy.debug(gen_model_dataset)

    # RUN TOPK
    # Get remote calculation of logs for topk stats (because we also calculate dockscore)
    if args.main.log_topk_freq > 0:
        log_top_k = LogTopKproc(args)

    # This logs statistics for MIN and MAX values for everything plotted (summary for wandb)
    log_summary = SummaryStats(log_wandb=do_plot)

    plot_train_k = ["train_loss_mean", "train_num_mol", "total_num_steps",
                    "train_num_sampled_mols", "num_sampled_mols",
                    "train_num_trajectories", "train_r_mean", "train_r_max", "FPS_mean"]
    plot_eval_k = ["eval_proxy_mean", "eval_proxy_min", "eval_proxy_cnt"]
    plot_top_k = ["top100_proxy_mean", "top100_proxy_min", "all_time_top10_score_mean",
                  "top100_proxy_count", "top100_count"]

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
    max_sampled_mols = getattr(gflow_args, "max_sampled", 0)

    # ==============================================================================================
    # -- Load GFLOW TWO
    gflow_args_two = args.gflow_two
    gflow_trainer_two = get_gflow_trainer(
        gflow_args_two, model=model_two, proxy=proxy_two, dataset=gen_model_dataset_two, do_save=False
    )

    proxy_two.setup_gflow_one(model_two, gen_model_dataset_two, gflow_trainer_two)

    # ==============================================================================================
    # -- Pre-load online mols
    pre_load_online_mols = getattr(gflow_dataset_args, "pre_load_online_mols", 0)
    if pre_load_online_mols > 0:
        from LambdaZero.examples.lightrl.utils.dataset_loaders import load_predocked_df
        pre_load_filter_cand = getattr(gflow_dataset_args, "pre_load_filter_cand", False)
        pre_load_add_max = getattr(gflow_dataset_args, "pre_load_add_max", 1.)

        df = load_predocked_df(Namespace(with_scores=True, dockscore_unique=True))

        if pre_load_filter_cand:
            # Expecting that proxy contains thresholds for qed and synth
            qed_th, synth_th = proxy.qed_th, proxy.synth_th
            df = df[(df.qed_score >= qed_th) & (df.synth_score >= synth_th)]

        # Debug might have a different num of blocks
        df = df[df.blockidxs.apply(max) < gen_model_dataset.mdp.num_blocks]

        # Filter by min/max num blocks
        num_blocks = df.blockidxs.apply(len)
        df = df[(num_blocks >= gflow_dataset_args.min_blocks) &
                (num_blocks <= gflow_dataset_args.max_blocks)]

        # Choose mols based on best dockscore ... which should be correlated with proxy
        df = df.sort_values("dockscore", ascending=True)
        num_best = int(pre_load_online_mols * pre_load_add_max)
        preloaded_mol = df.iloc[:num_best]
        if pre_load_online_mols - num_best > 0:
            other = df.iloc[num_best:].sample(pre_load_online_mols - num_best)
            preloaded_mol = pd.concat([preloaded_mol, other])

        # Calculate proxy score for mols
        mols = [gen_model_dataset.mdp.load(x) for x in preloaded_mol.to_dict(orient="records")]

        max_batch = 256
        online_mols = []
        rrrs = []
        proxy_scores = []
        for ibatch in range(0, len(mols), max_batch):  # run in max batch
            pmols = mols[ibatch: ibatch + max_batch]
            res_scores, infos = proxy(pmols)
            rrr = gen_model_dataset.r2r(dockscore=np.array(res_scores)).tolist()
            rrrs += rrr
            online_mols.extend(list(zip(rrr, rrr, pmols)))
            proxy_scores += [x["proxy"] for x in infos]

        for xonline in online_mols:
            gen_model_dataset._add_mol_to_online(*xonline)

        print(f"Added {len(online_mols)} mols with reward "
              f"mean {np.mean(rrrs)} | min {min(rrrs)} | max {max(rrrs)}")

        proxy_scores = sorted(proxy_scores)
        print(f"Histogram of proxy scores: {np.histogram(proxy_scores)}")
        print(f"Proxy min: {min(proxy_scores)} | Top100 mean: {np.mean(proxy_scores[:100])}")

    # ==============================================================================================
    num_sampled_mols = 0  # TODO no active learning implemented yet
    for i in range(num_outer_loop_iters):
        print(f"Starting step: {i}")

        # Initialize model and dataset for training generator
        # train model with with proxy
        # TODO Should have here some active learning code

        for epoch in range(num_iterations):
            training_metrics = gen_model_dataset.run_train_batch(gflow_trainer, epoch)
            train_data_stats, new_sampled_mol_infos = gen_model_dataset.get_stats(epoch)

            training_metrics.update(train_data_stats)
            trainer_num_smpl = gen_model_dataset.sampled_mols_cnt

            # ======================================================================================
            # Training error GFlowNet on data sampled from error GFlowNet dataset only!
            training_metrics_two = gen_model_dataset_two.run_train_batch(gflow_trainer_two, epoch)
            train_data_stats_two, new_sampled_mol_infos_two = gen_model_dataset_two.get_stats(epoch)
            training_metrics_two.update(train_data_stats_two)
            trainer_num_smpl_two = gen_model_dataset_two.sampled_mols_cnt

            # TODO Sample a batch of new molecules from model_two -> and add them to gen_model_dataset.online_mols
            # Will help with state space coverage of GFlowNet for better estimates of Q functions
            
            num_of_samples_two = 16
            new_mols_from_model_two = [
                gen_model_dataset_two.get_sample_model(explore=False) for _ in range(num_of_samples_two)
            ]  # type: List[BlockMoleculeDataExtended]
            samples = args.gflow_dataset.mdp.mols2batch([args.gflow_dataset.mdp.mol2repr(mol) for mol in new_mols_from_model_two])
            for xmol in samples:
                proxy_one_reward, _ = gen_model_dataset._get_reward(xmol)
                gen_model_dataset._add_mol_to_online(proxy_one_reward, proxy_one_reward, xmol, actually_add=True)

            # (for the beginning don't implement but Maybe pick the top only)

            # ======================================================================================

            # Every eval_freq number of training molecules add a batch of evaluation molecules
            if eval_freq > 0 and trainer_num_smpl >= last_eval_step + eval_freq:

                model.eval()
                eval_r, s, batch_metrics, log_metrics, sample_infos = \
                    sample_and_update_dataset_batch(
                        gflow_eval_args, model, gen_model_dataset, proxy, epoch
                    )
                model.train()

                new_sampled_mol_infos += sample_infos
                log_summary.update(log_metrics)
                training_metrics.update(log_metrics)
                last_eval_step += eval_freq

                # Plot evaluation stuff
                if do_plot:
                    wandb.log({"evalH_r_h": wandb.Histogram(gen_model_dataset.debug_r_counter)})
                    wandb.log({"evalH_s_h": wandb.Histogram(gen_model_dataset.debug_s_counter)})

                    debug_hist = batch_metrics.get("debug_hist", None)
                    if debug_hist is not None:
                        for hname, hinfo in debug_hist:
                            wandb.log({hname: hinfo})

                p_eval = trainer_num_smpl/(num_sampled_mols + len(new_sampled_mol_infos))*100
                training_metrics["p_eval_mols"] = p_eval

                print("-"*100, f"\n[E {epoch}] [EVAL] {plot_keys(plot_eval_k, log_metrics)}")
                print(f"[E {epoch}] [EVAL] Fraction of eval out of all sampled {p_eval:.2f} %")
                print("-"*100)

            num_sampled_mols += len(new_sampled_mol_infos)
            training_metrics["epoch"] = epoch
            training_metrics["outer_epoch"] = i
            training_metrics["num_sampled_mols"] = num_sampled_mols
            print(f"[E {epoch}] [TRAIN] {plot_keys(plot_train_k, training_metrics)}")

            if do_plot:
                wandb.log(training_metrics)

            if args.main.log_topk_freq > 0:
                # It starts calculating topk Automatically every args.main.log_topk_freq steps
                # Be sure that we send to collect all mol that want to be counted
                # We will calculate logging step based on Number of collected molecules
                log_top_k.collect(
                    new_sampled_mol_infos,
                    {"total_num_steps": training_metrics["total_num_steps"],
                    "step_num_sampled_mols": num_sampled_mols,
                     "train_num_sampled_mols": training_metrics["train_num_sampled_mols"]}
                )

                # Get async TopK logs (Has a different total_num_steps than previous logs)
                all_stats_top_k = log_top_k.get_stats()

                for stats_top_k in all_stats_top_k:
                    print("-"*100,
                          f"\n[E {epoch}] [TOPK] {plot_keys(plot_top_k, stats_top_k)}\n",
                          "-"*100)
                    log_summary.update(stats_top_k)
                    if do_plot:
                        wandb.log(stats_top_k)

            if max_sampled_mols > 0 and num_sampled_mols > max_sampled_mols:
                break

    gflow_trainer.stop_everything()
    gen_model_dataset.stop_samplers_and_join()

    # Sometimes wandb gets stuck; let's help him with everything we have :)
    print("IT'S WANDB`s fault"); sys.stdout.flush(); time.sleep(5); os._exit(0)
    sys.exit(); exit();  quit()


if __name__ == "__main__":

    from LambdaZero.examples.lightrl.utils.utils import parse_opts
    run(parse_opts())
