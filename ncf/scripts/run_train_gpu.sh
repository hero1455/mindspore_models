#!/bin/bash

if [ $# != 3 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_train_gpu.sh DATASET_PATH CKPT_FILE DEVICE_ID"
    echo "for example: bash scripts/run_train_gpu.sh /dataset_path /ncf.ckpt 0"
exit 1
fi

data_path=$1
ckpt_file=$2
export CUDA_VISIBLE_DEVICES=$3
python ./train.py \
    --data_path $data_path \
    --dataset 'ml-1m' \
    --train_epochs 25 \
    --batch_size 256 \
    --output_path './output/' \
    --checkpoint_path $ckpt_file  \
    --device_target=GPU \
    --device_id=0 \
    --num_parallel_workers=2 > train.log 2>&1 &
