#!/bin/bash

if [ $# != 2 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_transfer_ckpt_to_air.sh DATASET_PATH CKPT_FILE"
    echo "for example: bash scripts/run_transfer_ckpt_to_air.sh /dataset_path /ncf.ckpt"
exit 1
fi

data_path=$1
ckpt_file=$2
python ./src/export.py --data_path $data_path --dataset 'ml-1m'  --eval_batch_size 160000 --output_path './output/' --eval_file_name 'eval.log' --checkpoint_file_path $ckpt_file