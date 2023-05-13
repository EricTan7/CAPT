#!/bin/sh
#SBATCH --gpus=1

source activate ppt

for DATASET in imagenet ucf101
do
    for SHOTS in 16 8 4 2 1
    do
        WANDB_API_KEY=40afa4ca3f265a034bccdf4e176b2f2254081f21 WANDB_MODE=offline python train_wandb_iter.py \
        --dataset-config-file /data/run01/scz0bkt/code/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /data/run01/scz0bkt/code/Baseline/configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft.yaml \
        DATASET.NUM_SHOTS ${SHOTS} MODEL.BONDER.DEPTH 1
    done
done