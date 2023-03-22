for SHOTS in 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
    --root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/otuput/ \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/lpclip/rn50_ep50_batch32.yaml  \
    DATASET.NUM_SHOTS ${SHOTS} TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32
 done
