for seed in 2 3
do
    for shots in 16 8 4 2 1
    do
        python train.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32.yaml \
        DATASET.NUM_SHOTS ${shots} SEED ${seed}
    done
done