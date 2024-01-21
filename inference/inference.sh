#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python EAT/inference/inference.py  \
    --source_file='EAT/inference/test.wav' \
    --label_file='EAT/inference/labels.csv' \
    --model_dir='EAT' \
    --checkpoint_dir='/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EAT/EAT_finetuned_AS2M.pt' \
    --target_length=1024 \
    --top_k_prediction=12 \