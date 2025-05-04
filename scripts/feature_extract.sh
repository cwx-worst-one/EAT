#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

FRAMEWORK="huggingface"  # Options: "fairseq" or "huggingface"
CHECKPOINT_DIR="worstchan/EAT-base_epoch30_finetune_AS2M"  # HuggingFace repo or Fairseq checkpoint path
GRANULARITY="all"        # "all" (with CLS), "frame" (w/o CLS), or "utterance" (CLS only)
MODE="finetune"          # "pretrain" or "finetune"

python EAT/feature_extract/feature_extract.py  \
    --source_file='EAT/feature_extract/test.wav' \
    --target_file='EAT/feature_extract/test.npy' \
    --model_dir='EAT' \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --granularity=$GRANULARITY \
    --target_length=1024 \
    --mode=$MODE \
    --framework=$FRAMEWORK \
