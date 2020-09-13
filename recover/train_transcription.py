import torch
import torch.nn.functional as F
from recover.datasets.l1000_only_24hr_10mm import L1000TwentyFourHrTenMM
from recover.models.transcription_gen_mtx_factorization import TranscriptionGeneratorMatrixFactorizationModel
import os
from recover.models.utils import InnerProductPredictor, MLPPredictor, FingerprintMLPPredictor, FPCellLineMLPPredictor, \
    FilmFingerprintMLPPredictor, FilmMLPPredictor, CellLineMLPPredictor
from recover.utils import get_project_root
from torch.utils.data import TensorDataset, DataLoader
from hyperopt import hp
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import time
from ray import tune
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

def _get_datasets(data, config):
    full_dataset = TensorDataset(data.gene_expr, data.cell_id_int, data.cid)

    val_len = int(config['val_prop'] * len(full_dataset))
    test_len = int(config['test_prop'] * len(full_dataset))

    # Give leftover examples to train set
    train_len = len(full_dataset) - val_len - test_len

    train_set, val_set, test_set = torch.utils.data.random_split(full_dataset, [train_len, val_len, test_len])
    return train_set, val_set, test_set

def _get_loaders(config, train_set, val_set, test_set):
    train_loader = DataLoader(train_set,
                              batch_size=config['batch_size'],
                              pin_memory=(config['device'] == 'cpu'))

    val_loader = DataLoader(val_set,
                            batch_size=config['batch_size'],
                            pin_memory=(config['device'] == 'cpu'))

    test_loader = DataLoader(test_set,
                             batch_size=config['batch_size'],
                             pin_memory=(config['device'] == 'cpu'))

    return train_loader, val_loader, test_loader

def _get_normalizers(train_set, val_set, test_set):
    result = {}
    for set_name in ["train", "val", "test"]:
        curr_set = locals()["%s_set" % set_name]
        result[set_name] = MeanVarianceNormalizer((curr_set[:][0].mean(axis=0),
                                                   curr_set[:][0].var(axis=0)))

    return result['train'], result['val'], result['test']

def run_epoch(loader, model, normalizer, optim, is_train):
    model.train() if is_train else model.eval()

    loss_sum      = 0
    epoch_targets = []
    epoch_preds   = []

    # Context manager for if we are in train mode and need to pass grad
    none_ctx = contextmanager(lambda: iter([None]))()
    with none_ctx if is_train else torch.no_grad():
        for batch in loader:
            y, cell_line_idx, drug_idx = batch
            y_hat = model(cell_line_idx, drug_idx)
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


class TranscriptionGenTrainer(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["device"] = self.device

        dataset = config["dataset"](transform=config["transform"], pre_transform=config["pre_transform"])

        self.data = dataset[0].to(self.device)

        torch.manual_seed(config["seed"])

        train_set, val_set, test_set = _get_datasets(self.data, config)
        self.train_loader, self.valid_loader, self.test_loader = _get_loaders(self.config, train_set,
                                                                              val_set, test_set)

        self.train_norm, self.valid_norm, self.test_norm = _get_normalizers(train_set, val_set, test_set)

        self.valid_var = val_set[:][0].var()
        self.test_var = test_set[:][0].var()

        # Initialize model and optimizer
        self.model = config["model"](self.data, config).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

    def _train(self):
        all_scores = {}
        for set_name in ["train", "valid", "test"]:
            is_train = set_name == "train"
            optim = self.optim if is_train else None

            scores = run_epoch(getattr(self, "%s_loader" % set_name), self.model,
                               getattr(self, "%s_norm" % set_name), optim, is_train)

            # Check if len(scores) > 0 as if the dictionary is empty then there
            # was no data for this set
            if not is_train and len(scores) > 0:
                var = getattr(self, "%s_var" % set_name)
                scores["r_squared"] = (var - scores["mse"]) / var

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
    config = {
        "trainer": TranscriptionGenTrainer,
        "trainer_config": {
            "transform": None,
            "pre_transform": None,
            "seed": 1,
            "val_prop": 0.2,
            "test_prop": 0.,
            "model": TranscriptionGeneratorMatrixFactorizationModel,
            "dataset": L1000TwentyFourHrTenMM,
        },
        "summaries_dir": summaries_dir,
        "memory": 1800,
        "asha_metric": "valid_mse",
        "asha_mode": "min",
        "asha_max_t": 1000,
        "checkpoint_freq": 20,
        "stop": {"training_iteration": 1000},
        "checkpoint_at_end": True,
        "resources_per_trial": {},#"cpu": 10, "gpu": 1},
        "name": "TranscriptionGenBaseline"
    }

    asha_scheduler = ASHAScheduler(
	time_attr='training_iteration',
	metric=config['asha_metric'],
	mode=config['asha_mode'],
	max_t=config['asha_max_t'],
	grace_period=10,
	reduction_factor=3,
	brackets=1
    )

    search_space = {
        "lr": hp.loguniform("lr", -16.118095651, -5.52146091786),
        "num_mtxs": hp.quniform("num_mtxs", 1, 12, 1),
        "mtx_inner_dim": hp.quniform("mtx_inner_dim", 64, 512, 1),
        "alternate_mtx_types": hp.choice("alternate_mtx_types", [False, True]),
        "dropout_prob": hp.uniform("dropout_prob", .0, .7),
        "use_film": hp.choice("use_film", [False, True]),
    }

    current_best_params = [
        {
            "lr": 1e-4,
            "num_mtxs": 5,
            "mtx_inner_dim": 150,
            "alternate_mtx_types": 0,
            "use_film": 1,
            "dropout_prob": .3,
        }
    ]

    search_alg = HyperOptSearch(
        search_space,
        metric=config['asha_metric'],
        mode=config['asha_mode'],
        points_to_evaluate=current_best_params
    )


    analysis = tune.run(
        config["trainer"],
        name=config["name"],
        config=config["trainer_config"],
        stop=config["stop"],
        resources_per_trial=config["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=config["checkpoint_at_end"],
        local_dir=config["summaries_dir"],
        checkpoint_freq=config["checkpoint_freq"],
        scheduler=asha_scheduler,
        search_alg=search_alg,
    )


