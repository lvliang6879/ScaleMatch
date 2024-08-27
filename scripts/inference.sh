#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these arguments if you want to try other datasets or methods
# dataset: ['pascal', 'cityscapes', 'coco', 'ade20k']
# method: ['prevmatch', 'supervised']
# exp: just for specifying the 'save_path'
dataset='pascal'
config=configs/${dataset}.yaml

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env inference.py \
    --dataset=$dataset \
    --config=$config \
    --ckpt-path=$3 
    --port $2
