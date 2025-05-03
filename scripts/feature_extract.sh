#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

FRAMEWORK="huggingface"     # "fairseq" or "huggingface"
CHECKPOINT_DIR="worstchan/EAT-base_epoch30_finetune_AS2M"  # Path to the checkpoint file ( xxx.pt for fairseq, repo id for huggingface)
GRANULARITY="all"        # "all", "frame", or "utterance"

python EAT/feature_extract/feature_extract.py  \
    --source_file='EAT/feature_extract/test.wav' \
    --target_file='EAT/feature_extract/test.npy' \
    --model_dir='EAT' \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --granularity=$GRANULARITY \
    --target_length=1024 \
    --mode='finetune' \
    --framework=$FRAMEWORK \

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# You should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.

# 3 ways to extract features
# all: all frame features including the cls token
# frame: all frame features excluding the cls token
# utterance: only the cls token feature.

# 2 mode of checkpoints: (EAT) pretrain or (EAT) finetune
# 2 framework: fairseq or huggingface