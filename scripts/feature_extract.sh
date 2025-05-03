#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

FRAMEWORK="huggingface"  # Options: "fairseq" or "huggingface"
CHECKPOINT_DIR="worstchan/EAT-base_epoch30_finetune_AS2M"  # HuggingFace repo or Fairseq checkpoint path
GRANULARITY="all"        # "all" (with CLS), "frame" (w/o CLS), or "utterance" (CLS only)

python EAT/feature_extract/feature_extract.py  \
    --source_file='EAT/feature_extract/test.wav' \
    --target_file='EAT/feature_extract/test.npy' \
    --model_dir='EAT' \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --granularity=$GRANULARITY \
    --target_length=1024 \
    --mode='finetune' \
    --framework=$FRAMEWORK \


# Notes:
# - Recommended target_length = 1024 for 10-second clips (512 for 5-second).
# - Ensure target_length is a multiple of 16 due to CNN encoder constraints.
# - Choose granularity based on task:
#     all       → frame-level features incl. CLS
#     frame     → frame-level features excl. CLS
#     utterance → only CLS token (utterance-level embedding)
# - Two modes of checkpoints: (EAT) pretrain or (EAT) finetune.
