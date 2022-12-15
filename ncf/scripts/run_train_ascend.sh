#!/bin/bash

if [ $# != 2 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_train_ascend.sh DATASET_PATH CKPT_FILE"
    echo "for example: bash scripts/run_train_ascend.sh /dataset_path /ncf.ckpt"
exit 1
fi

data_path=$1
ckpt_file=$2
python ./train.py --data_path $data_path --dataset 'ml-1m'  --train_epochs 25 --batch_size 256 \
    --output_path './output/' --checkpoint_path $ckpt_file --device_target=Ascend > train.log 2>&1 &
