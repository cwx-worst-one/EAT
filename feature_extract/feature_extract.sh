#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python feature_extract.py  \
    --source_file='/hpc_stor03/sjtu_home/wenxi.chen/EAT/feature_extract/test.wav' \
    --target_file='/hpc_stor03/sjtu_home/wenxi.chen/EAT/feature_extract/test.npy' \
    --model_dir='/hpc_stor03/sjtu_home/wenxi.chen/EAT' \
    --checkpoint_dir='/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EAT/checkpoint10.pt' \
    --granularity='frame' \
    --target_length=1024 \
