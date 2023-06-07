for SHOTS in 16 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_l14.yaml \
    DATASET.NUM_SHOTS ${SHOTS}
done