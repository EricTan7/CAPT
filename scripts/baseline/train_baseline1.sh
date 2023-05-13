for DATASET in imagenet
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=7 python train_wandb_iter.py \
        --root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/output/${DATASET}/s{SHOTS} \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft.yaml \
        DATASET.NUM_SHOTS ${SHOTS} DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001
    done
done