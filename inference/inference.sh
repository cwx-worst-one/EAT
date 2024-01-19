#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python inference/inference.py  \
    --source_file='/hpc_stor03/sjtu_home/wenxi.chen/EAT/inference/test.wav' \
    --label_file='/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/balanced_train/label_descriptors.csv' \
    --model_dir='/hpc_stor03/sjtu_home/wenxi.chen/EAT' \
    --checkpoint_dir='/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EAT/EAT_finetuned_AS2M.pt' \
    --target_length=1024 \
    --top_k_prediction=12 \