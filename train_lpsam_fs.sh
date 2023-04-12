for SHOTS in 16 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
    --root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpsam/output/vit_b/imagenet/s${SHOTS} \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/lpsam/vit_b_batch32.yaml  \
    DATASET.NUM_SHOTS ${SHOTS} TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32 \
    OPTIM.LR 0.001
done