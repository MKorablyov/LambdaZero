ROOT_PATH=/Users/bruno/LambdaZero/summaries/env3d/dataset/from_cluster/RUN4/data/
DATA_FILE_NAME=combined_dataset
CONFIG=input.json

python ../train_model.py --root_path $ROOT_PATH --data_file_name $DATA_FILE_NAME --config=$CONFIG
