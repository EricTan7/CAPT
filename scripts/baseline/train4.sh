CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.4 MODEL.LORA.RANK 2 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \
CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.6 MODEL.LORA.RANK 2 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \
CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 2 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \
CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 2. MODEL.LORA.RANK 2 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \
CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.4 MODEL.LORA.RANK 4 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \
CUDA_VISIBLE_DEVICES=5 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.6 MODEL.LORA.RANK 4 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \
CUDA_VISIBLE_DEVICES=6 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 4 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \
CUDA_VISIBLE_DEVICES=7 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 2. MODEL.LORA.RANK 4 \
    OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5 & \