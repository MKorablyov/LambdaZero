from LambdaZero.examples.drug_comb.dataset.new_drugcomb_data import NewDrugComb
import os
from LambdaZero.utils import get_external_dirs
import time
from ray import tune
import ray
from sklearn.linear_model import LinearRegression
import torch
import pandas as pd
import torch.nn.functional as F

X_IDX = 0
Y_IDX = 1

def train(data, model):
    if data[X_IDX].shape[0] == 0:
        return {
            "rmse": 1e8,
            "mse": 1e8,
            "mae": 1e8,
        }

    model.fit(data[X_IDX], data[Y_IDX])

    preds = torch.tensor(model.predict(data[X_IDX]), dtype=torch.float)
    return {
        "rmse": torch.sqrt(F.mse_loss(preds, data[Y_IDX]) + 1e-8).item(),
        "mse": F.mse_loss(preds, data[Y_IDX]).item(),
        "mae": F.l1_loss(preds, data[Y_IDX]).item(),
    }

def test(data, model):
    if data[X_IDX].shape[0] == 0:
        return {
            "rmse": 1e8,
            "mse": 1e8,
            "mae": 1e8,
        }

    preds = torch.tensor(model.predict(data[X_IDX]), dtype=torch.float)

    return {
        "rmse": torch.sqrt(F.mse_loss(preds, data[Y_IDX]) + 1e-8).item(),
        "mse": F.mse_loss(preds, data[Y_IDX]).item(),
        "mae": F.l1_loss(preds, data[Y_IDX]).item(),
    }

class CellLineRegressor(tune.Trainable):
    def _setup(self, config):
        dataset = NewDrugComb().to_cell_line(config['cell_line_name'])

        train_idx, val_idx, _ = dataset.random_split(.0, config['val_prob'])

        self.train_data = dataset[train_idx]
        self.val_data = dataset[val_idx]

        self.model = LinearRegression()

    def _train(self):
        train_scores = train(self.train_data, self.model)
        eval_scores = test(self.val_data, self.model)

        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)

        num_examples = 0
        for item in self.train_data + self.val_data:
            num_examples += item.shape[0]

        scores["num_examples"] = num_examples

        return scores

    def _save(self, checkpoint_dir):
        pass

    def _restore(self, checkpoint_path):
        pass

def _get_cell_line_names(datasets_dir):
    raw_dir = os.path.join(datasets_dir, 'NewDrugComb/raw')

    cell_line_path = os.path.join(raw_dir, 'cell_lines_corr2.csv')
    if not os.path.exists(cell_line_path):
        summary_csv = pd.read_csv(
            os.path.join(datasets_dir, 'NewDrugComb/raw/summary_table_v1.4.csv')
        )

        summary_csv = summary_csv.groupby('cell_line_name').filter(lambda x: len(x) >= 1)
        summary_csv['cell_line_name'].drop_duplicates().to_csv(cell_line_path)

    return pd.read_csv(cell_line_path)['cell_line_name'].tolist()

datasets_dir, _, summaries_dir =  get_external_dirs()
config = {
    "trainer": CellLineRegressor,
    "trainer_config": {
        "val_prob": 0.2,
        "cell_line_name": tune.grid_search(_get_cell_line_names(datasets_dir)),
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 200,
    "stop": {"training_iteration": 1},
    "checkpoint_at_end": False,
    "name": None
}

if __name__ == "__main__":
    ray.init()

    time_to_sleep = 2
    print("Sleeping for %d seconds" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    config["name"] = "CellLineTaskAll"
    analysis = tune.run(
        config["trainer"],
        name=config["name"],
        config=config["trainer_config"],
        stop=config["stop"],
        #resources_per_trial=config["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=config["checkpoint_at_end"],
        local_dir=config["summaries_dir"],
        checkpoint_freq=config["checkpoint_freq"]
    )
