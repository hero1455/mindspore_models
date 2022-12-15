#!/bin/bash

if [ $# != 3 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_eval_ascend.sh DATASET_PATH CKPT_FILE DEVICE_ID"
    echo "for example: bash scripts/run_eval_ascend.sh /dataset_path NCF-25_19418.ckpt 0"
exit 1
fi

data_path=$1
ckpt_file=$2
export DEVICE_ID=$3
python ./eval.py --data_path $data_path --dataset 'ml-1m'  --eval_batch_size 160000 \
    --output_path './output/' --eval_file_name 'eval.log' --checkpoint_file_path $ckpt_file \
    --device_target=Ascend --device_id $DEVICE_ID > eval.log 2>&1 &
