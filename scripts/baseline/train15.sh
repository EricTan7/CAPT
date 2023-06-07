# for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet sun397
for DATASET in oxford_pets stanford_cars oxford_flowers eurosat
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=5 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_t5/vit_b16_t5_base.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done