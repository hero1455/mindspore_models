#!/bin/bash

if [ $# -lt 1 ]; then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_distribute_train.sh RANK_TABLE_FILE [DATA_PATH]"
    echo "for example: bash scripts/run_distribute_train.sh /path/hccl.json /dataset_path"
    echo "It is better to use absolute path."
    echo "=============================================================================================================="
    exit 1
fi

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

python3 ${BASE_PATH}/ascend_distributed_launcher/get_distribute_pretrain_cmd.py \
    --run_script_path=${BASE_PATH}/../train.py \
    --hccl_config_dir=$1 \
    --hccl_time_out=600 \
    --args=" --data_path=$2 \
        --dataset='ml-1m' \
        --train_epochs=50 \
        --output_path='./output/' \
        --eval_file_name='eval.log' \
        --checkpoint_path='./checkpoint/' \
        --device_target='Ascend'" \
    --cmd_file=distributed_cmd.sh

bash distributed_cmd.sh
