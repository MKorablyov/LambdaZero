import numpy as np
import torch
from torch.nn import functional as F


def class_and_angle_loss(
    block_predictions, block_targets, angle_predictions, angle_targets, loss_type='cos'
):
    """
    calculate losses for block predictions (class) and angle predictions

    Args:
        block_predictions (torch.Tensor): class predictions logits. size: (batchsize, number of classes)
        block_targets (torch.Tensor): class target as int. size (batchsize)
        angle_predictions (torch.Tensor): vector prediction for the angle. Will be converted to angle
            size: (batchsize, 2)
        angle_targets (torch.Tensor): angle targets in radian between 0 and 2 pi. Value of -1 means an invalid angle.
            size: (batchsize)
        loss_type (str, optional): if cos, the loss is the L2 norm of cos & sin of prediction and target.
            If angle, the loss is the L2 distance between the prediction angle and the target.
            Defaults to cos

    Returns:
        torch.Tensor: cross entropy for the class. size: (1,)
        torch.Tensor: mse for the sin and cos of the angle. size: (1,)
        torch.Tensor: mae for sin and cos of the angle. size (1,)
        int: number of valid angles
    """
    assert loss_type in ['cos', 'angle'], 'loss mode should be cos or angle'

    # prediction over classes in a straight-forward cross-entropy
    class_loss = F.cross_entropy(block_predictions, block_targets)

    # for the angle, we convert the outputs to sin / cos representation
    # sin = u / \sqrt{u² + v²}
    # cos = v / \sqrt{u² + v²}
    
    if loss_type == 'cos':
        # get denominator
        norm = torch.norm(angle_predictions, dim=-1)
        # take max between norm and a small value to avoid division by zero
        norm = torch.max(norm, 1e-6 * torch.ones_like(norm))
        # norm is a (batchsize) tensor. convert to (batchsize, 2)
        norm = norm.unsqueeze(-1).repeat(1, 2)
        angle_predictions = (
            angle_predictions / norm
        )  # the idiom angle_predictions /= norm leads to a pytorch runtime error
        # angle_predictions[:, 0] is sin, [:, 1] is cos
        # now, convert the ground truth
        sin_target = torch.sin(angle_targets)
        cos_target = torch.cos(angle_targets)
        angle_target_sincos = torch.stack([sin_target, cos_target], dim=-1)
        # loss for the angle is the MSE
        angle_loss = F.mse_loss(angle_target_sincos, angle_predictions, reduction="none")
        angle_mae = F.l1_loss(angle_target_sincos, angle_predictions, reduction="none")
    
        # sum over last dimension, aka sin and cos
        angle_loss = torch.sum(angle_loss, dim=-1)
        angle_mae = torch.sum(angle_mae, dim=-1)

    elif loss_type == 'angle':
        angle_predictions = torch.atan2(angle_predictions[:, 0], angle_predictions[:, 1])
        # loss is the MSE for the angle
        angle_loss = F.mse_loss(angle_targets, angle_predictions, reduction="none")
        angle_loss_2pi = F.mse_loss(angle_targets + 2 * np.pi, angle_predictions, reduction="none")
        angle_loss_m2pi = F.mse_loss(angle_targets - 2 * np.pi, angle_predictions, reduction="none")
        angle_loss = torch.min(angle_loss, angle_loss_2pi)
        angle_loss = torch.min(angle_loss, angle_loss_m2pi)
        angle_mae = F.l1_loss(angle_targets, angle_predictions, reduction="none")
        angle_mae_2pi = F.l1_loss(angle_targets + 2 * np.pi, angle_predictions, reduction="none")
        angle_mae_m2pi = F.l1_loss(angle_targets - 2 * np.pi, angle_predictions, reduction="none")
        angle_mae = torch.min(angle_mae, angle_mae_2pi)
        angle_mae = torch.min(angle_mae, angle_mae_m2pi)

    # create a mask of 0 where angle_target is invalid (-1), and 1 elsewhere
    mask = torch.where(
        angle_targets >= 0,
        torch.ones_like(angle_targets),
        torch.zeros_like(angle_targets),
    )
    num_elem = torch.sum(mask)

    # calculate the mean over valid elements only
    angle_loss = torch.sum(angle_loss * mask) / max(num_elem, 1)
    angle_mae = torch.sum(angle_mae * mask) / max(num_elem, 1)

    return class_loss, angle_loss, angle_mae, num_elem