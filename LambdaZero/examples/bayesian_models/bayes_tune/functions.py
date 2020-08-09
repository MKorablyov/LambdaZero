import time, random
import numpy as np
from scipy.special import logsumexp
from sklearn import linear_model
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader


def _epoch_metrics(epoch_targets_norm, epoch_logits, normalizer, scope):
    epoch_targets = normalizer.itfm(epoch_targets_norm)
    epoch_preds = normalizer.itfm(epoch_logits)
    metrics = {}
    #metrics["loss"] = metrics["loss"] / epoch_targets.shape[0] todo
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()
    # ranking
    ranked_targets = epoch_targets[np.argsort(epoch_targets)]
    predsranked_targets = epoch_targets[np.argsort(epoch_preds)]
    metrics["top15_regret"] = np.median(predsranked_targets[:15]) - np.median(ranked_targets[:15])
    metrics["top50_regret"] = np.median(predsranked_targets[:50]) - np.median(ranked_targets[:50])
    if scope is not None: metrics = dict([(scope + "_" + k, v) for k, v in metrics.items()])
    return metrics

def get_tau(config, N):
    tau = (1 - config["model_config"]["drop_prob"]) * (config["lengthscale"]**2) / (2 * N * config["lambda"])
    return tau

def _log_lik(y, Yt_hat, config, N):
    "computes log likelihood"
    # ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat) ** 2., 0)
    # - np.log(T)
    # - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau))
    tau = get_tau(config, N)
    ll = logsumexp(-0.5 * tau * (y[None] - Yt_hat) ** 2., 0)
    ll -= np.log(Yt_hat.shape[0])
    ll -= 0.5 * np.log(2 * np.pi)
    ll += 0.5 * np.log(tau)
    return ll


def train_epoch(loader, model, optimizer, device, config, scope):
    model.train()
    epoch_targets_norm = []
    epoch_logits = []

    for bidx,data in enumerate(loader):
        data = data.to(device)
        targets = getattr(data, config["target"])

        optimizer.zero_grad()
        logits = model(data, do_dropout=True)
        targets_norm = config["normalizer"].tfm(targets)
        reg_loss = config['lambda'] * torch.stack([(p ** 2).sum() for p in model.parameters()]).sum()
        loss = F.mse_loss(logits, targets_norm) + reg_loss
        loss.backward()
        optimizer.step()

        epoch_targets_norm.append(targets_norm.detach().cpu().numpy())
        epoch_logits.append(logits.detach().cpu().numpy())

    epoch_targets_norm = np.concatenate(epoch_targets_norm,0)
    epoch_logits = np.concatenate(epoch_logits, 0)
    scores = _epoch_metrics(epoch_targets_norm, epoch_logits, config["normalizer"], scope)
    return scores


def sample_logits(loader, model, device, config, num_samples, do_dropout):
    sample_logits = []
    for i in range(num_samples):
        epoch_logits = []
        for bidx, data in enumerate(loader):
            data = data.to(device)
            logit = model(data, do_dropout=do_dropout)
            epoch_logits.append(logit.detach().cpu().numpy())
        sample_logits.append(np.concatenate(epoch_logits, 0))
    return np.stack(sample_logits,0)


def sample_targets(loader, config):
    epoch_targets = [getattr(d, config["target"]).cpu().numpy() for d in loader.dataset]
    norm_targets = config["normalizer"].tfm(np.concatenate(epoch_targets,0))
    return norm_targets


def eval_epoch(loader, model, device, config, scope):
    logits = sample_logits(loader, model, device, config, 1, False)[0]
    norm_targets = sample_targets(loader, config)
    scores = _epoch_metrics(norm_targets, logits, config["normalizer"], scope)
    return scores


def eval_mcdrop(loader, model, device, config, N, scope):
    norm_targets = sample_targets(loader, config)
    logits = sample_logits(loader, model, device, config, num_samples=config["T"], do_dropout=True)
    ll = _log_lik(norm_targets, logits, config, N).mean()
    shuff_targets = np.array(sorted(norm_targets, key=lambda k: random.random()))
    shuf_ll = _log_lik(shuff_targets, logits, config, N).mean()
    return {scope + "_ll":ll, scope + "_shuff_ll":shuf_ll}


def train_mcdrop(train_loader, val_loader, model, device, config, optim, iteration):
    N = len(train_loader.dataset)
    train_scores = config["train_epoch"](train_loader, model, optim, device, config, "train")
    val_scores = config["eval_epoch"](val_loader, model, device, config, "val")
    scores = {**train_scores, **val_scores}

    if iteration % config["uncertainty_eval_freq"] == 1:
        _scores = eval_mcdrop(val_loader, model, device, config, N, "val")
        scores = {**scores, **_scores}
    return scores


def mcdrop_mean_variance(train_loader, loader, model, device, config):
    # \mean{t in T} (\tau^-1 + y_hat_t^2) - \mean_{t in T}(y_hat_t)^2
    N = len(train_loader.dataset)
    Yt_hat = sample_logits(loader, model, device, config, config["T"], do_dropout=True)
    tau = get_tau(config, N)
    sigma_sqr = 1. / tau
    var = (sigma_sqr + Yt_hat ** 2).mean(0) - Yt_hat.mean(0) ** 2
    return Yt_hat.mean(0), var


def bayesian_ridge(train_x, val_x, train_targets_norm, val_targets_norm, config):
    clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)
    clf.fit(train_x, train_targets_norm)
    train_logits = clf.predict(train_x)
    val_logits, val_std = clf.predict(val_x, return_std=True)
    train_scores = _epoch_metrics(train_targets_norm, train_logits, config["normalizer"], "train_ridge")
    val_scores = _epoch_metrics(val_targets_norm, val_logits, config["normalizer"], "val_ridge")
    ll = -0.5 * np.mean(np.log(2 * np.pi * val_std ** 2) + ((val_targets_norm - val_logits) ** 2 / val_std ** 2))
    val_scores["val_ll"] = ll
    return {**train_scores, **val_scores}


def sample_embeds(loader, model, device, config):
    epoch_embeds = []
    for bidx, data in enumerate(loader):
        data = data.to(device)
        embeds = model.get_embed(data, do_dropout=False, drop_p=config["model_config"]["drop_prob"])
        epoch_embeds.append(embeds.detach().cpu().numpy())
    epoch_embeds = np.concatenate(epoch_embeds,axis=0)
    return epoch_embeds


def eval_mpnn_brr(train_loader, val_loader, model, device, config, N):
    # todo(maksym) I am not sure what is the best way to keep order
    train_loader = DataLoader(train_loader.dataset, batch_size=config["b_size"])
    val_loader = DataLoader(val_loader.dataset, batch_size=config["b_size"])

    train_targets_norm = sample_targets(train_loader, config)
    val_targets_norm = sample_targets(val_loader, config)
    train_embeds = sample_embeds(train_loader, model, device, config)
    val_embeds = sample_embeds(val_loader, model, device, config)
    scores = bayesian_ridge(train_embeds, val_embeds, train_targets_norm, val_targets_norm, config)
    return scores

def train_mpnn_brr(train_loader, val_loader, model, device, config, optim, iteration):
    N = len(train_loader.dataset)
    train_scores = config["train_epoch"](train_loader, model, optim, device, config, "train")
    val_scores = config["eval_epoch"](val_loader, model, device, config, "val")
    scores = {**train_scores, **val_scores}

    if iteration % config["uncertainty_eval_freq"] == 1:
        _scores = eval_mpnn_brr(train_loader, val_loader, model, device, config, N)
        scores = {**scores, **_scores}
    return scores


def mpnn_brr_mean_variance(train_loader, loader, model, device, config):
    # train model on the train_set (should be fast)
    train_loader = DataLoader(train_loader.dataset, batch_size=config["b_size"])
    train_embeds = sample_embeds(train_loader, model, device, config)
    train_targets_norm = sample_targets(train_loader, config)
    embeds = sample_embeds(loader, model, device, config)

    clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)
    clf.fit(train_embeds, train_targets_norm)
    mean, std = clf.predict(embeds, return_std=True)
    return mean, std