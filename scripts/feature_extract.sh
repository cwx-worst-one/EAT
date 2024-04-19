#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python EAT/feature_extract/feature_extract.py  \
    --source_file='EAT/feature_extract/test.wav' \
    --target_file='EAT/feature_extract/test.npy' \
    --model_dir='EAT' \
    --checkpoint_dir='/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EAT/checkpoint10.pt' \
    --granularity='frame' \
    --target_length=1024 \
    --mode='pretrain'

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# You should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.

# 3 ways to extract features
# all: all frame features including the cls token
# frame: all frame features excluding the cls token
# utterance: only the cls token feature.

# 2 mode of checkpoints: (EAT) pretrain or (EAT) finetune