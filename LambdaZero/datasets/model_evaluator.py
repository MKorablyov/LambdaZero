import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


class MoleculeModelEvaluator:

    def __init__(self, device):
        self.device = device

    def _model_step(self, batch, model):
        batch = batch.to(self.device)
        y = batch.dockscore
        y_hat = model.forward(batch)
        return y, y_hat

    def get_actuals_and_predicted(self, dataloader: DataLoader, model: nn.Module):
        model.eval()

        list_actuals = []
        list_predicted = []
        with torch.no_grad():
            for batch in dataloader:
                y, y_hat = self._model_step(batch, model)
                list_actuals.extend(list(y.numpy()))
                list_predicted.extend(list(y_hat.numpy()))

        return np.array(list_actuals), np.array(list_predicted)

