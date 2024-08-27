#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these arguments if you want to try other datasets or methods
# dataset: ['pascal', 'cityscapes', 'coco', 'ade20k']
# method: ['prevmatch', 'supervised']
# exp: just for specifying the 'save_path'
dataset='pascal'
method='ScaleMatch'
exp='r101'
split=732

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
