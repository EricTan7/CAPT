CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_textsup.yaml \
DATASET.NUM_SHOTS 1 OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5 & \

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/ucf101.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_textsup.yaml \
DATASET.NUM_SHOTS 1 OPTIM.MAX_ITER 38400 OPTIM.LR 5e-5 & \