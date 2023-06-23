#!/usr/bin/env bash

DATASET=$1
SHOTS=$2
ALPHA=$3
RANK=$4

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/${DATASET}.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} MODEL.LORA.ALPHA ${ALPHA} MODEL.LORA.RANK ${RANK} \
    OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5 \