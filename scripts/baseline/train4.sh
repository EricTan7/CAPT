for SHOTS in 16 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_l14_wo_caption.yaml \
    DATASET.NUM_SHOTS ${SHOTS}
done