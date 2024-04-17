#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python EAT/evaluation/eval.py  \
    --label_file='EAT/inference/labels.csv' \
    --eval_dir='/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/AS20K' \
    --model_dir='EAT' \
    --checkpoint_dir='/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EATs/finetuning/as20k_epoch30/checkpoint_last.pt' \
    --target_length=1024 \
    --device='cuda' \
    --batch_size=32

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make evaluation well even given truncated audio clips.