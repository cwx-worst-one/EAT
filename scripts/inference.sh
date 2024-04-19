#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python EAT/inference/inference.py  \
    --source_file='EAT/inference/test.wav' \
    --label_file='EAT/inference/labels.csv' \
    --model_dir='EAT' \
    --checkpoint_dir='/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EAT/EAT_finetuned_AS2M.pt' \
    --target_length=1024 \
    --top_k_prediction=12 \

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make inference well even given truncated audio clips.