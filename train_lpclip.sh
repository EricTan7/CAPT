for SHOTS in 16 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=5 python train_wandb.py \
    --root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/output/vit_b16/imagenet/s${SHOTS} \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/lpclip/vit_b16_batch32.yaml  \
    DATASET.NUM_SHOTS ${SHOTS} TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32 \
    OPTIM.LR 0.001
done

for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=5 python train_wandb.py \
        --root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/output/vit_b16/${DATASET}/s${SHOTS} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/lpclip/vit_b16_batch32.yaml  \
        DATASET.NUM_SHOTS ${SHOTS} TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        OPTIM.LR 0.002
     done
done