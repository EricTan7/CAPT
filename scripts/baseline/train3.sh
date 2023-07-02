CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_text.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 25600 OPTIM.LR 5e-5 & \

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_text_cross.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 25600 OPTIM.LR 5e-5 & \

CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/oxford_pets.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_text.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 25600 OPTIM.LR 2e-5 & \

CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/oxford_pets.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_text_cross.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 25600 OPTIM.LR 2e-5 & \