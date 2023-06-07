# fgvc_aircraft oxford_pets stanford_cars dtd eurosat oxford_flowers
for DATASET in fgvc_aircraft oxford_pets stanford_cars
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_l14.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done