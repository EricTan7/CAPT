CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/outuput/ \
--dataset-config-file configs/datasets/food101.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32

CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/outuput/ \
--dataset-config-file configs/datasets/oxford_flowers.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 8 TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32

CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/outuput/ \
--dataset-config-file configs/datasets/oxford_pets.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32

CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/outuput/ \
--dataset-config-file configs/datasets/ucf101.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32

CUDA_VISIBLE_DEVICES=6 python train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/output/imagenet/s4 \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 4 TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32