#!/usr/bin/env bash

DATASET=$1
SHOTS=$2
FIX_ITER=$3
LR=$4

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/${DATASET}.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16_fixedfirst.yaml \
    DATASET.NUM_SHOTS ${SHOTS} MODEL.LORA.ALPHA 0.6 MODEL.LORA.RANK 4 \
    OPTIM.MAX_ITER 19200 TRAIN.FIX_EPOCH ${FIX_ITER} OPTIM.LR ${LR}