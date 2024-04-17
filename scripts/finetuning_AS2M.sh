#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    checkpoint.save_dir=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/finetune_test \
    checkpoint.restore_file=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/finetune_test/checkpoint_last.pt \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=96 \
    task.data=/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/AS2M \
    task.h5_format=true \
    task.AS2M_finetune=true \
    task.weights_file=/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/AS2M/weight_train_all.csv \
    task.target_length=1024 \
    task.roll_aug=true \
    model.model_path=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EAT/checkpoint10.pt \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \