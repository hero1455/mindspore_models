#!/bin/bash

if [ $# != 1 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_download_dataset.sh DATASET_PATH"
    echo "for example: bash scripts/run_download_dataset.sh /dataset_path"
exit 1
fi

data_path=$1
python ./src/movielens.py --data_path $data_path --dataset 'ml-1m'
