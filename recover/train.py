import torch
from recover.datasets.drugcomb_data import DrugComb, DrugCombNoPPI
from recover.datasets.drugcomb_score_data import DrugCombScore, DrugCombScoreNoPPI
from recover.datasets.drugcomb_score_l1000_data import DrugCombScoreL1000, DrugCombScoreL1000NoPPI
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
import ray

########################################################################################################################
# Epoch loops
########################################################################################################################


def train_epoch(data, loader, model, optim):
    model.train()
    epoch_loss = 0
    num_batches = len(loader)

    for i, drug_drug_batch in enumerate(loader):
        optim.zero_grad()

        out = model.forward(data, drug_drug_batch)
        loss = model.loss(out, drug_drug_batch)

        loss.backward()
        optim.step()

        epoch_loss += loss.item() * data.response_std**2

    print('Mean train loss: {:.4f}'.format(epoch_loss / num_batches))

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches}


def eval_epoch(data, loader, model):
    model.eval()
    epoch_loss = 0
    r_squared = 0
    num_batches = len(loader)

    with torch.no_grad():
        for i, drug_drug_batch in enumerate(loader):
            out = model.forward(data, drug_drug_batch)

            loss = model.loss(out, drug_drug_batch)
            epoch_loss += loss.item() * data.response_std**2

            # Explained variance
            var = drug_drug_batch[3].var().item()
            r_squared += (var - loss.item())/var

    print('Mean valid loss: {:.4f}'.format(epoch_loss / num_batches))

    return {"loss_sum": epoch_loss, "loss_mean": epoch_loss / num_batches, "r_squared": r_squared / num_batches}


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

        self.data = dataset[0].to(self.device)

        # If a score dataset is used, we have to specify the target
        if "target" in config.keys():
            possible_target_dicts = {"css": self.data.ddi_edge_css,
                                     "bliss": self.data.ddi_edge_bliss,
                                     "zip": self.data.ddi_edge_zip,
                                     "hsa": self.data.ddi_edge_hsa,
                                     "loewe": self.data.ddi_edge_loewe}
            self.data.ddi_edge_response = possible_target_dicts[config["target"]]

        # Normalize mean and variance
        self.data.ddi_edge_response -= self.data.ddi_edge_response.mean()
        self.data.response_std = self.data.ddi_edge_response.std().item()
        self.data.ddi_edge_response /= self.data.ddi_edge_response.std()

        torch.manual_seed(config["seed"])

        train_idxs, val_idxs, test_idxs = dataset.random_split(config["test_set_prop"], config["val_set_prop"])

        # Train loader
        train_ddi_dataset = TensorDataset(self.data.ddi_edge_idx[:, train_idxs].T,
                                          self.data.ddi_edge_classes[train_idxs], self.data.ddi_edge_attr[train_idxs],
                                          self.data.ddi_edge_response[train_idxs])

        self.train_loader = DataLoader(train_ddi_dataset,
                                       batch_size=config["batch_size"],
                                       pin_memory=(self.device == 'cpu'))

        # Valid loader
        valid_ddi_dataset = TensorDataset(self.data.ddi_edge_idx[:, val_idxs].T,
                                          self.data.ddi_edge_classes[val_idxs], self.data.ddi_edge_attr[val_idxs],
                                          self.data.ddi_edge_response[val_idxs])

        self.valid_loader = DataLoader(valid_ddi_dataset, batch_size=config["batch_size"],
                                       pin_memory=(self.device == 'cpu'))

        # Initialize model and optimizer
        self.model = config["model"](self.data, config).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.data, self.train_loader, self.model, self.optim)
        eval_scores = self.eval_epoch(self.data, self.valid_loader, self.model)

        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)

        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == '__main__':
    ray.init(num_cpus=40)

    time_to_sleep = 30
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
            "model": tune.grid_search([MLPBaselineFPExpr, MLPBaselineFPProt]),
            "dataset": DrugCombScoreL1000NoPPI,
            "target": "css",  # tune.grid_search(["css", "bliss", "zip", "loewe", "hsa"]),
            "fp_bits": 1024,
            "fp_radius": 4,
            "ppi_confidence_thres": 0,
            "train_epoch": train_epoch,
            "eval_epoch": eval_epoch,
            "batch_size": 128,  # tune.grid_search([1024, 2048]),
        },
        "summaries_dir": summaries_dir,
        "memory": 1800,
        "checkpoint_freq": 20,
        "stop": {"training_iteration": 30},
        "checkpoint_at_end": False,
        "resources_per_trial": {"cpu": 10, "gpu": 1},
        "name": "ExpressionBaseline"
    }

    analysis = tune.run(
        configuration["trainer"],
        name=configuration["name"],
        config=configuration["trainer_config"],
        stop=configuration["stop"],
        resources_per_trial=configuration["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=configuration["checkpoint_at_end"],
        local_dir=configuration["summaries_dir"],
        checkpoint_freq=configuration["checkpoint_freq"]
    )

    # trainer = GiantGraphTrainer(configuration["trainer_config"])
    # trainer.train()
