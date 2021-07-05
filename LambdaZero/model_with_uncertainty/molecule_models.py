from torch.utils.data import DataLoader

import numpy as np

from LambdaZero.contrib.modelBO.molecule_models import MolMCDropGNN
from LambdaZero.contrib.data import ListGraphDataset
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior


def get_mcdrop_var_batch(model, data, num_mc_samples):
    y_hat_mc = []
    for i in range(num_mc_samples):
        y_hat_epoch = []
        y_hat_batch = model(data, do_dropout=True)[:, 0]
        y_hat_epoch.append(y_hat_batch.detach().cpu().numpy())
        y_hat_mc.append(np.concatenate(y_hat_epoch, 0))
    y_hat_mc = np.stack(y_hat_mc, 1)
    return y_hat_mc.mean(1), y_hat_mc.var(1)


class MolMCDropGNN2(MolMCDropGNN):
    def __init__(self, train_epochs, batch_size, mpnn_config, lr, transform, num_mc_samples, log_epoch_metrics, device,
                   logger):
        MolMCDropGNN.__init__(self,  train_epochs, batch_size, mpnn_config, lr, transform, num_mc_samples, log_epoch_metrics, device,
                   logger)

    def posterior(self, x, observation_noise=False):
        mean_m, variance_m = self.get_mean_and_variance(x)
        if observation_noise:
            pass

        mvn = MultivariateNormal(mean_m.squeeze(), torch.diag(variance_m.squeeze() + 1e-6))
        return GPyTorchPosterior(mvn)

    def get_embed(self, x):
        graphs = [m["mol_graph"] for m in x]
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list)
        embed = None
        for batch in dataloader:
            batch.to(self.device)
            y_hat_batch = self.model.get_embed(batch, do_dropout=False)
            if embed is None:
                embed = y_hat_batch.detach().cpu()
            else:
                embed = torch.cat((embed, y_hat_batch.detach().cpu()), dim=0)

        return embed