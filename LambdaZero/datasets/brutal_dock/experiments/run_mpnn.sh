# This script shows how to drive the execution of an experiment.
# To run locally, substitute "LAMBDA_ZERO_BASE_PATH" with a meaningful path.

LAMBDA_ZERO_BASE_PATH="/Users/bruno/PycharmProjects/LambdaZero/"
INPUT=./example_mpnn_input.json
SCRIPT=mpnn_experiment.py

DATA_DIRECTORY=$LAMBDA_ZERO_BASE_PATH/Datasets/brutal_dock/d4/raw/
OUTPUT_DIRECTORY=$LAMBDA_ZERO_BASE_PATH/results/
WORKING_DIRECTORY=$LAMBDA_ZERO_BASE_PATH/LambdaZero/datasets/brutal_dock/experiments/data/
TRACKING_URI=$LAMBDA_ZERO_BASE_PATH/mlruns/

python $SCRIPT --input=$INPUT  \
               --working_directory=$WORKING_DIRECTORY \
               --output_directory=$OUTPUT_DIRECTORY \
               --data_directory=$DATA_DIRECTORY \
               --tracking_uri=$TRACKING_URI 
