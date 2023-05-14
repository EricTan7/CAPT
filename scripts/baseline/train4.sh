#!/bin/sh
#SBATCH --gpus=1

source activate ppt

# for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet
for DATASET in dtd imagenet
do
    for SHOTS in 16 8 4 2 1
    do
        WANDB_API_KEY=40afa4ca3f265a034bccdf4e176b2f2254081f21 WANDB_MODE=offline python train_wandb_iter_val.py \
        --dataset-config-file /data/run01/scz0bkt/code/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /data/run01/scz0bkt/code/Baseline/configs/trainers/Baseline_sattn/vit_b16.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done