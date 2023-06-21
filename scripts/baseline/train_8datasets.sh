 #!/usr/bin/env bash

SHOTS=$1

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/caltech101.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} & \
CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/dtd.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} & \
CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/fgvc_aircraft.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} & \
CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/food101.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} & \
CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/eurosat.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} & \
CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/ucf101.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} & \
CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/oxford_flowers.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} & \
CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/oxford_pets.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS}