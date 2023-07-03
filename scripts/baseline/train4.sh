CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/caltech101.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_textsup.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 38400 OPTIM.LR 2e-5 & \

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_textsup.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5 & \

CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/dtd.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_textsup.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 25600 OPTIM.LR 5e-5 & \

CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/eurosat.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_textsup.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 25600 OPTIM.LR 5e-5 & \