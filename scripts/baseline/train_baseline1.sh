for DATASET in imagenet oxford_pets
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
        --root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/output/${DATASET}/s{SHOTS} \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_aug.yaml  \
        DATASET.NUM_SHOTS ${SHOTS} TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32
     done
done