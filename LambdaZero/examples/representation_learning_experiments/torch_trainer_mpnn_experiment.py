from pathlib import Path

import ray
import torch
from ray import tune
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import NUM_SAMPLES
from torch import nn

from LambdaZero import ROOT_DIR
from LambdaZero.representation_learning.dataloader_utils import get_geometric_dataloaders
from LambdaZero.representation_learning.datasets import D4GeometricMoleculesDataset
from LambdaZero.representation_learning.models.message_passing_model import MessagePassingNet
from LambdaZero.representation_learning.parameter_inputs import CONFIG_KEY, NON_CONFIG_KEY, PATHS_KEY, \
    TRAINING_PARAMETERS_KEY, MODEL_PARAMETERS_KEY, read_configuration_file, EXECUTION_FILENAME_KEY
from LambdaZero.utils import get_external_dirs


class PytorchGeometricTrainingOperator(TrainingOperator):
    target_name = "gridscore"

    def train_batch(self, batch, batch_info):
        # Create non_blocking tensors for distributed training
        if torch.cuda.is_available():
            batch = batch.cuda(non_blocking=True)

        target = batch[self.target_name]
        # Compute output.
        with self.timers.record("fwd"):
            output = self.model(batch)
            loss = self.criterion(output, target)

        # Compute gradients in a backward pass.
        with self.timers.record("grad"):
            self.optimizer.zero_grad()
            loss.backward()

        # Call step of optimizer to update model params.
        with self.timers.record("apply"):
            self.optimizer.step()

        return {"train_loss": loss.item(), NUM_SAMPLES: batch.num_graphs}

    def validate_batch(self, batch, batch_info):
        if torch.cuda.is_available():
            batch = batch.cuda(non_blocking=True)

        # compute output
        target = batch[self.target_name]

        with self.timers.record("eval_fwd"):
            output = self.model(batch)
            loss = self.criterion(output, target)

        return {"val_loss": loss.item(), NUM_SAMPLES: batch.num_graphs}


path_of_this_file = Path(__file__).resolve()

json_file_path = ROOT_DIR.joinpath("LambdaZero/examples/representation_learning_experiments/configurations/example_mpnn_input.json")


def model_creator(config):
    model_class = MessagePassingNet
    input_and_run_config = config["input_and_run_config"]
    model_parameters = input_and_run_config[CONFIG_KEY][MODEL_PARAMETERS_KEY]
    return model_class.create_model_for_training(model_parameters)


def data_creator(config):
    dataset_class = D4GeometricMoleculesDataset

    random_seed = config["random_seed"]
    input_and_run_config = config["input_and_run_config"]

    paths_dict = input_and_run_config[NON_CONFIG_KEY][PATHS_KEY]
    training_parameters = input_and_run_config[CONFIG_KEY][TRAINING_PARAMETERS_KEY]

    data_dir = Path(paths_dict["data_directory"])
    work_dir = Path(paths_dict["working_directory"])

    dataset = dataset_class.create_dataset(root_dir=work_dir,
                                           original_raw_data_dir=data_dir)

    training_dataloader, validation_dataloader, test_dataloader = \
        get_geometric_dataloaders(dataset, training_parameters, random_seed)

    return training_dataloader, validation_dataloader


def optimizer_creator(model, config):
    input_and_run_config = config["input_and_run_config"]
    training_parameters = input_and_run_config[CONFIG_KEY][TRAINING_PARAMETERS_KEY]
    return torch.optim.Adam(model.parameters(), lr=training_parameters["learning_rate"])


if __name__ == '__main__':

    datasets_dir, programs_dir, summaries_dir = get_external_dirs()

    config_parameters = read_configuration_file(json_file_path)

    debug_path = Path(summaries_dir).joinpath("debug")

    paths = {"tracking_uri": str(debug_path.joinpath("tracking_uri")),
             "working_directory": str(debug_path.joinpath("working")),
             "output_directory": str(debug_path.joinpath("output")),
             "data_directory": str(Path(datasets_dir).joinpath("brutal_dock/d4/raw/"))
             }

    non_config_parameters = {PATHS_KEY: paths,
                             EXECUTION_FILENAME_KEY: str(path_of_this_file.relative_to(ROOT_DIR))}

    input_and_run_config = {CONFIG_KEY: config_parameters,
                            NON_CONFIG_KEY: non_config_parameters}

    config = {"random_seed": 0, "input_and_run_config": input_and_run_config}

    #ray.init()
    ray.init(local_mode=True)

    trainer = TorchTrainer.as_trainable(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=nn.MSELoss,
        config=config,
        use_gpu=False,
        num_workers=1,
        use_tqdm=True,
        training_operator_cls=PytorchGeometricTrainingOperator
    )

    tuning_config = dict(model_parameters=dict(gcn_size=tune.grid_search([16, 32, 64])))

    analysis = tune.run(trainer,
                        config=tuning_config,
                        stop={"training_iteration": 20},
                        num_samples=1,
                        checkpoint_at_end=True,
                        local_dir=summaries_dir,
                        checkpoint_freq=100)
