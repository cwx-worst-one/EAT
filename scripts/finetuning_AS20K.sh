#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    checkpoint.save_dir=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/finetune_as20k \
    checkpoint.restore_file=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/finetune_as20k/checkpoint_last.pt \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=48 \
    task.data=/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/AS20K \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EAT/checkpoint10.pt \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \