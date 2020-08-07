import numpy as np
import torch

from LambdaZero.examples.env3d.loss import class_and_angle_loss


def train_epoch(loader, model, optimizer, device, config):
    """
    function training a model over an epoch. Metrics are computed as well

    Args:
        loader (torch_geometric.dataInMemoryDataset: training dataloader
        model (nn.Module): pytorch model
        optimizer (torch.optim): optimizer to train the model
        device (torch.device): device where the model / data will be processed
        config (dict): other parameters

    Returns:
        dict: metrics for the epoch. Includes:
            loss for block and angle predictions combined
            block_loss for block predictions only (cross-entropy)
            angle_loss for loss predictions only (MSE on sin and cos)
            accuracy for block predictions
            RMSE for sin/cos of angle
            MAE for sin/cos of angle
    """
    model.train()

    metrics = {}

    for data in loader:

        data = data.to(device)

        class_target = data.attachment_block_class
        angle_target = data.attachment_angle

        optimizer.zero_grad()
        # model outputs 2 tensors:
        # 1) size (batch, nclass) for block prediction
        # 2) size (batch, 2) for angle prediction. u and v should be combined to get sin / cos of angle
        class_predictions, angle_predictions = model(data)

        # prediction over classes in a straight-forward cross-entropy
        class_loss, angle_loss, angle_mae, n_angle = class_and_angle_loss(
            class_predictions, class_target, angle_predictions, angle_target
        )

        loss = class_loss + config.get("angle_loss_weight", 1) * angle_loss

        loss.backward()
        optimizer.step()

        # log loss information
        # global cumulative losses on this epoch
        metrics["loss"] = metrics.get("loss", 0) + loss.item() * data.num_graphs
        metrics["block_loss"] = (
            metrics.get("block_loss", 0) + class_loss.item() * data.num_graphs
        )
        metrics["angle_loss"] = (
            metrics.get("angle_loss", 0) + angle_loss.item() * data.num_graphs
        )

        # let's log some more informative metrics as well
        # for prediction, accuracy is good to know
        _, predicted = torch.max(class_predictions, dim=1)
        correct = (predicted == class_target).sum().item()
        # consider the accuracy as a rolling average. We have to adjust the previously calculated accuracy by the number of elements in it
        metrics["block_accuracy"] = (
            metrics.get("block_accuracy", 0) * metrics.get("n_blocks", 0) + correct
        ) / max(1, metrics.get("n_blocks", 0) + data.num_graphs)
        metrics["n_blocks"] = metrics.get("n_blocks", 0) + data.num_graphs

        # for the angle, we want the RMSE and the MAE, both as rolling averages as well. Beware the mask for invalid angles here
        metrics["angle_rmse"] = (
            metrics.get("angle_rmse", 0) * metrics.get("n_angles", 0)
            + angle_loss.item() * n_angle
        ) / max(1, metrics.get("n_angles", 0) + n_angle)
        # we have to do the same for the MAE
        metrics["angle_mae"] = (
            metrics.get("angle_mae", 0) * metrics.get("n_angles", 0) + angle_mae.item() * n_angle
        ) / max(1, metrics.get("n_angles", 0) + n_angle)

    # RMSE and MAE are possibly on gpu. Move to cpu and convert to numpy values (non-tensor)
    # also take the squareroot of the MSE to get the actual RMSE
    metrics["angle_rmse"] = np.sqrt(metrics["angle_rmse"].cpu().item())
    metrics["angle_mae"] = metrics["angle_mae"].cpu().item()

    return metrics


def eval_epoch(loader, model, device, config):
    """
    evaluate a model and return metrics

    Args:
        loader (torch_geometric.dataInMemoryDataset: training dataloader
        model (nn.Module): pytorch model
        device (torch.device): device where the model / data will be processed
        config (dict): other parameters

    Returns:
        dict: metrics for the epoch. Includes:
            loss for block and angle predictions combined
            block_loss for block predictions only (cross-entropy)
            angle_loss for loss predictions only (MSE on sin and cos)
            accuracy for block predictions
            RMSE for sin/cos of angle
            MAE for sin/cos of angle`
    """
    model.eval()
    metrics = {}

    for data in loader:
        data = data.to(device)

        class_target = data.attachment_block_class
        angle_target = data.attachment_angle

        # model outputs 2 tensors:
        # 1) size (batch, nclass) for block prediction
        # 2) size (batch, 2) for angle prediction. u and v should be combined to get sin / cos of angle
        class_predictions, angle_predictions = model(data)

        # prediction over classes in a straight-forward cross-entropy
        class_loss, angle_loss, angle_mae, n_angle = class_and_angle_loss(
            class_predictions, class_target, angle_predictions, angle_target
        )

        loss = class_loss + config.get("angle_loss_weight", 1) * angle_loss

        # log loss information
        # global cumulative losses on this epoch
        metrics["loss"] = metrics.get("loss", 0) + loss.item() * data.num_graphs
        metrics["block_loss"] = (
            metrics.get("block_loss", 0) + class_loss.item() * data.num_graphs
        )
        metrics["angle_loss"] = (
            metrics.get("angle_loss", 0) + angle_loss.item() * data.num_graphs
        )

        # let's log some more informative metrics as well
        # for prediction, accuracy is good to know
        _, predicted = torch.max(class_predictions, dim=1)
        correct = (predicted == class_target).sum().item()
        # consider the accuracy as a rolling average. We have to adjust the previously calculated accuracy by the number of elements in it
        metrics["block_accuracy"] = (
            metrics.get("block_accuracy", 0) * metrics.get("n_blocks", 0) + correct
        ) / max(1, metrics.get("n_blocks", 0) + data.num_graphs)
        metrics["n_blocks"] = metrics.get("n_blocks", 0) + data.num_graphs

        # for the angle, we want the RMSE and the MAE, both as rolling averages as well. Beware the mask for invalid angles here
        metrics["angle_rmse"] = (
            metrics.get("angle_rmse", 0) * metrics.get("n_angles", 0)
            + angle_loss.item() * n_angle
        ) / max(1, metrics.get("n_angles", 0) + n_angle)
        # we have to do the same for the MAE
        metrics["angle_mae"] = (
            metrics.get("angle_mae", 0) * metrics.get("n_angles", 0) + angle_mae.item()
        ) / max(1, metrics.get("n_angles", 0) + n_angle)

    # RMSE and MAE are possibly on gpu. Move to cpu and convert to numpy values (non-tensor)
    # also take the squareroot of the MSE to get the actual RMSE
    metrics["angle_rmse"] = np.sqrt(metrics["angle_rmse"].cpu().item())
    metrics["angle_mae"] = metrics["angle_mae"].cpu().item()

    return metrics