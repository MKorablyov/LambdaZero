# This script shows how to drive the execution of the train_model.py script.
# To run locally, substitute "ROOT_PATH" with a meaningful path.

ROOT_PATH=/Users/bruno/LambdaZero/summaries/env3d/dataset/from_cluster/RUN4/data/
DATA_FILE_NAME=combined_dataset
CONFIG=example_input.json

python ../train_model.py --root_path $ROOT_PATH --data_file_name $DATA_FILE_NAME --config=$CONFIG
