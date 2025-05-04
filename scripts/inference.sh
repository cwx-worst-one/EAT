#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

FRAMEWORK="huggingface"  # Options: "fairseq" or "huggingface"
CHECKPOINT_DIR="worstchan/EAT-base_epoch30_finetune_AS2M"  # HuggingFace repo or Fairseq checkpoint path


python EAT/inference/inference.py  \
    --source_file='EAT/inference/test.wav' \
    --label_file='EAT/inference/labels.csv' \
    --model_dir='EAT' \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --target_length=1024 \
    --top_k_prediction=12 \
    --framework=$FRAMEWORK \


# Notes:
# - Recommended target_length = 1024 for 10-second audio (100Hz mel-spectrogram).
# - EAT can still perform robust inference on clips with different lengths, though performance may vary.