import torch
import torch.nn.functional as F
from recover.datasets.drugcomb_data import DrugComb, DrugCombNoPPI
from recover.datasets.drugcomb_score_data import DrugCombScore, DrugCombScoreNoPPI
from recover.datasets.drugcomb_score_l1000_data import DrugCombScoreL1000, DrugCombScoreL1000NoPPI
from recover.models.gnn import GNN
from recover.models.multi_message_gcn import GiantGraphGCN
from recover.models.baselines import Dummy, ConcentrationOnlyBaseline, ScalarProdBaselineFP, MLPBaselineFP, \
    MLPBaselineFPProt, FilmMLPBaselineFPProt, MLPBaselineExpr, MLPBaselineFPExpr
from recover.models.message_conv_layers import ThreeMessageConvLayer, ProtDrugMessageConvLayer, \
    DummyMessageConvLayer
import os
from recover.models.utils import InnerProductPredictor, MLPPredictor, FingerprintMLPPredictor, FPCellLineMLPPredictor, \
    FilmFingerprintMLPPredictor, FilmMLPPredictor, CellLineMLPPredictor
from recover.utils import get_project_root
from torch.utils.data import TensorDataset, DataLoader
import time
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
from contextlib import contextmanager
import ray

class MeanVarianceNormalizer:
    def __init__(self, mean_and_variance):
        self.mean = mean_and_variance[0]
        self.variance = mean_and_variance[1]

    def tfm(self, x):
        "normalize x"
        x_norm = (x - self.mean) / self.variance
        return x_norm

    def itfm(self, x_norm):
        "unnormalize x"
        x = (x_norm * self.variance) + self.mean
        return x

########################################################################################################################
# Epoch loops
########################################################################################################################

def _get_loaders(data, batch_size, device, train_idxs, val_idxs, test_idxs):
    result = {}
    for set_name in ["train", "val", "test"]:
        idxs = locals()["%s_idxs" % set_name]

        dataset = TensorDataset(data.ddi_edge_idx[:, idxs].T,
                                data.ddi_edge_classes[idxs],
                                data.ddi_edge_attr[idxs],
                                data.ddi_edge_response[idxs])

        result[set_name] = DataLoader(dataset,
                                      batch_size=batch_size,
                                      pin_memory=(device == 'cpu'))

    return result["train"], result["val"], result["test"]

def _get_normalizers(data, train_idxs, val_idxs, test_idxs):
    result = {}
    for set_name in ["train", "val", "test"]:
        idxs = locals()["%s_idxs" % set_name]
        result[set_name] = MeanVarianceNormalizer((data.ddi_edge_response[idxs].mean(),
                                                   data.ddi_edge_response[idxs].var()))

    return result["train"], result["val"], result["test"]

def run_epoch(data, loader, model, normalizer, optim, is_train):
    RESPONSE_IDX = 3
    model.train() if is_train else model.eval()

    loss_sum      = 0
    epoch_targets = []
    epoch_preds   = []

    # Context manager for if we are in train mode and need to pass grad
    none_ctx = contextmanager(lambda: iter([None]))()
    with none_ctx if is_train else torch.no_grad():
        for batch in loader:
            y = batch[RESPONSE_IDX]
            y_hat = model(data, batch)
            loss = F.mse_loss(normalizer.tfm(y), y_hat)

            if is_train:
                loss.backward()
                optim.step()
                optim.zero_grad()

            loss_sum += loss.item()
            epoch_targets.append(y.detach().cpu())
            epoch_preds.append(normalizer.itfm(y_hat).detach().cpu())

    scores = {}
    if len(epoch_targets) > 0 and len(epoch_preds) > 0:
        epoch_targets = torch.cat(epoch_targets)
        epoch_preds = torch.cat(epoch_preds)

        scores = {
            "loss": loss_sum / epoch_targets.shape[0],
            "mae": F.l1_loss(epoch_targets, epoch_preds).item(),
            "mse": F.mse_loss(epoch_targets, epoch_preds).item(),
            "rmse": torch.sqrt(F.mse_loss(epoch_targets, epoch_preds)).item(),
        }

    return scores

########################################################################################################################
# Main training loop
########################################################################################################################


class GiantGraphTrainer(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["device"] = self.device

        dataset = config["dataset"](transform=config["transform"], pre_transform=config["pre_transform"],
                                    fp_bits=config["fp_bits"], fp_radius=config["fp_radius"],
                                    ppi_confidence_thres=config["ppi_confidence_thres"])

        # Need to call this here so that the edge index properties are propagated later
        self.data = dataset[0].to(self.device)

        # If a score dataset is used, we have to specify the target
        if "target" in config.keys():
            possible_target_dicts = {"css": self.data.ddi_edge_css,
                                     "bliss": self.data.ddi_edge_bliss,
                                     "zip": self.data.ddi_edge_zip,
                                     "hsa": self.data.ddi_edge_hsa,
                                     "loewe": self.data.ddi_edge_loewe}

            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        torch.manual_seed(config["seed"])

        train_idxs, val_idxs, test_idxs = dataset.random_split(config["test_set_prop"], config["val_set_prop"])
        self.train_loader, self.valid_loader, self.test_loader = _get_loaders(self.data, config["batch_size"],
                                                                              self.device, train_idxs, val_idxs,
                                                                              test_idxs)

        # Assign idxs to the data object here since if we did earlier
        # there'd be an error data.to() is called
        self.data.train_edge_index = self.data.ddi_edge_idx[:, train_idxs]
        self.data.test_edge_index = self.data.ddi_edge_idx[:, test_idxs]
        self.data.val_edge_index = self.data.ddi_edge_idx[:, val_idxs]

        self.train_norm, self.valid_norm, self.test_norm = _get_normalizers(self.data, train_idxs,
                                                                            val_idxs, test_idxs)

        self.valid_var = self.data.ddi_edge_response[val_idxs].var()
        self.test_var = self.data.ddi_edge_response[test_idxs].var()

        # Initialize model and optimizer
        self.model = config["model"](self.data, config).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

    def _train(self):
        all_scores = {}
        for set_name in ["train", "valid", "test"]:
            is_train = set_name == "train"
            optim = self.optim if is_train else None

            scores = run_epoch(self.data, getattr(self, "%s_loader" % set_name),
                               self.model, getattr(self, "%s_norm" % set_name),
                               optim, is_train)

            if not is_train and len(scores) > 0:
                var = getattr(self, "%s_var" % set_name)
                scores["%s_r_squared" % set_name] = (var - scores["mse"]) / var

            all_scores.update({"%s_%s" % (set_name, key): val for key, val in scores.items()})

        return all_scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == '__main__':
    ray.init(num_cpus=40)

    time_to_sleep = 5
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    summaries_dir = get_project_root()
    configuration = {
        "trainer": GiantGraphTrainer,
        "trainer_config": {
            "transform": None,
            "pre_transform": None,
            "seed": 1,  # tune.grid_search([1, 2, 3]),
            "aggr": "concat",
            "gcn_dropout_rate": .1,
            "gcn_channels": [1024, 256, 256, 256],
            "gnn_lyr_type": "GCNWithAttention",
            "num_residual_gcn_layers": 1,
            "linear_dropout_rate": .2,
            "linear_channels": [64, 32, 16, 1],
            "num_relation_lin_layers": 2,
            # "scalar_layers": [100, 100, 10],  # For scalar baselines
            "mlp_layers": [4096, 2048, 1024, 1],  #tune.grid_search([[1], [1024, 1]]),  # For MLP baselines
            "conv_layer": ThreeMessageConvLayer,
            "attention": True,  # tune.grid_search([False, True]),
            "attention_rank": tune.grid_search([64, 128, 256, 512]),
            "prot_emb_dim": 1024,  # For GCN
            "residual_layers_dim": 1024,  # For GCN
            'num_res_layers': 0,  # tune.grid_search([0, 1]),  # For GCN
            "predictor": CellLineMLPPredictor,
            "predictor_layers": [1024, 512, 256, 1],  # tune.grid_search([[2048, 1024, 1], [4096, 2048, 1024, 1]]),
            "val_set_prop": 0.2,
            "test_set_prop": 0.,
            "lr": 1e-4,  # tune.grid_search([1e-4, 5e-4]),
            "model": GNN,
            "dataset": DrugCombScore,
            "target": "css",  # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
            "fp_bits": 1024,
            "fp_radius": 4,
            "ppi_confidence_thres": 0,
            "batch_size": 128,  # tune.grid_search([1024, 2048]),
        },
        "summaries_dir": summaries_dir,
        "memory": 1800,
        "checkpoint_freq": 20,
        "stop": {"training_iteration": 30},
        "checkpoint_at_end": False,
        "resources_per_trial": {},#"cpu": 10, "gpu": 1},
        "name": "ExpressionBaseline",
        "asha_metric": "eval_mse",
        "asha_mode": "min",
        "asha_max_t": 100,
    }

    asha_scheduler = ASHAScheduler(
	time_attr='training_iteration',
	metric=configuration['asha_metric'],
	mode=configuration['asha_mode'],
	max_t=configuration['asha_max_t'],
	grace_period=10,
	reduction_factor=3,
	brackets=1
    )

    search_space = {
        "lr": hp.loguniform("lr", -16.118095651, -5.52146091786),
        "rank": hp.quniform("rank", 50, 300, 1),
        "batch_size": hp.choice("batch_size", [256, 512]),
    }

    current_best_params = [
        {
            "lr": 1e-4,
            "rank": 128,
            "batch_size": 0,
        }
    ]

    search_alg = HyperOptSearch(
        search_space,
        metric=configuration['asha_metric'],
        mode=configuration['asha_mode'],
        points_to_evaluate=current_best_params
    )


    analysis = tune.run(
        configuration["trainer"],
        name=configuration["name"],
        config=configuration["trainer_config"],
        stop=configuration["stop"],
        resources_per_trial=configuration["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=configuration["checkpoint_at_end"],
        local_dir=configuration["summaries_dir"],
        checkpoint_freq=configuration["checkpoint_freq"],
        scheduler=asha_scheduler,
        search_alg=search_alg,
    )

