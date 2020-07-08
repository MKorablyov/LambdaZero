# This script shows how to drive the execution of an experiment.
# To run locally, substitute "LAMBDA_ZERO_BASE_PATH" with a meaningful path.

export ORION_DB_ADDRESS=/home/simon/Work/research/lambdazero/LambdaZero/LambdaZero/datasets/brutal_dock/experiments/orion_wd/orion.pkl
export ORION_DB_TYPE='pickleddb'
export ORION_DB_NAME='brutaldock1'

LAMBDA_ZERO_BASE_PATH="/home/simon/Work/research/lambdazero/LambdaZero/"
INPUT=orion_mpnn_input.json
SCRIPT=mpnn_experiment.py

DATA_DIRECTORY=$LAMBDA_ZERO_BASE_PATH/Datasets/brutal_dock/d4/raw/
OUTPUT_DIRECTORY=$LAMBDA_ZERO_BASE_PATH/results/
WORKING_DIRECTORY=$LAMBDA_ZERO_BASE_PATH/LambdaZero/datasets/brutal_dock/experiments/data/
TRACKING_URI=$LAMBDA_ZERO_BASE_PATH/mlruns/

orion -v hunt -n brutaldoc_test1 --config orion_config.yaml ./$SCRIPT --config=$INPUT  \
              --working_directory=$WORKING_DIRECTORY \
              --output_directory=$OUTPUT_DIRECTORY \
              --data_directory=$DATA_DIRECTORY \
              --tracking_uri=$TRACKING_URI
