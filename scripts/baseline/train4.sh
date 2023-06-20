for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet sun397
do
    for shots in 16 8 4 2 1
    do
        for seed in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=0 python train.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_t5/vit_b16_t5_base.yaml \
            SEED ${seed} DATASET.NUM_SHOTS ${shots}
        done
    done
done