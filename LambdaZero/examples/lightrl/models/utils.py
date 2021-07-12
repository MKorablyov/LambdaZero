import torch


def one_hot(idx, device, max_cnt):
    hot = torch.zeros(len(idx), max_cnt, device=device).scatter_(1, idx.flatten().unsqueeze(1), 1.)
    return hot


def rsteps_rcond_conditioning(out, inputs, max_steps):
    data = inputs.mol_graph

    rsteps = one_hot(inputs.r_steps[data.batch], inputs.r_steps.device, max_steps)
    out = torch.cat([out, rsteps, inputs.rcond.float()[data.batch].unsqueeze(1)], dim=1)
    return out