#!/bin/sh
#SBATCH --gpus=1

source activate ppt

WANDB_API_KEY=40afa4ca3f265a034bccdf4e176b2f2254081f21 WANDB_MODE=offline python train_wandb_iter.py \
--root /data/run01/scz0bkt/datasets/recognition/ --seed 1 \
--output-dir /data/run01/scz0bkt/datasets/recognition/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft_mul.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 100