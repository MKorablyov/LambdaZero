from LambdaZero.datasets.brutal_dock import ROOT_DIR, RESULTS_DIR
from LambdaZero.datasets.brutal_dock.parameter_inputs import RUN_PARAMETERS_KEY, TRAINING_PARAMETERS_KEY, \
    MODEL_PARAMETERS_KEY, write_configuration_file


def get_example_config():
    run_parameters = dict(experiment_name='Message Passing Neural Network',
                          run_name="debugging",
                          )

    training_parameters = dict(num_epochs=100,
                               num_workers=4,
                               batch_size=4096,
                               learning_rate=1e-4,
                               train_fraction=0.8,
                               validation_fraction=0.1)

    model_parameters = dict(name="MPNN",
                            gcn_size=128,
                            edge_hidden=128,
                            gru_out=128,
                            gru_layers=1,
                            linear_hidden=128)

    config = {RUN_PARAMETERS_KEY: run_parameters,
              TRAINING_PARAMETERS_KEY: training_parameters,
              MODEL_PARAMETERS_KEY: model_parameters}

    return config


if __name__ == '__main__':
    json_config_path = str(RESULTS_DIR.joinpath("input.json"))
    config_dict = get_example_config()
    write_configuration_file(json_config_path, config_dict)
