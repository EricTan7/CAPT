#!/usr/bin/env bash

DATASET=$1
SHOTS=$2
ITER=$3
LR=$4

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/${DATASET}.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_caption/vit_b16_se_cross.yaml \
    DATASET.NUM_SHOTS ${SHOTS} OPTIM.MAX_ITER ${ITER} OPTIM.LR ${LR}

