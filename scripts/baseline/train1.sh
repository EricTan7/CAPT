for DATASET in imagenet_wval
do
    for shots in 16 8 4 2 1
    do
        for seed in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=0 python train.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
            SEED ${seed} DATASET.NUM_SHOTS ${shots}
        done
    done
done