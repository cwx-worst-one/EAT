#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name pretraining_AS2M \
    common.user_dir=EAT \
    checkpoint.save_dir=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/test \
    checkpoint.restore_file=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/test/checkpoint_last.pt \
    distributed_training.distributed_world_size=4 \
    dataset.batch_size=12 \
    task.data=/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/unbalanced_train \
    task.h5_format=True \