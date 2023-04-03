# stanford_cars food101 caltech101 ucf101 oxford_pets oxford_flowers fgvc_aircraft dtd eurosat
for DATASET in imagenet
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=6 python train.py \
        --root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/output/${DATASET}/s${SHOTS} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/Baseline/rn50_ep50_batch32_imagenet.yaml  \
        DATASET.NUM_SHOTS ${SHOTS} TRAIN.PRINT_FREQ 3 DATALOADER.TRAIN_X.BATCH_SIZE 32 \
        TRAIN.TEST_FREQ 50 TRAIN.SAVE_FREQ 50
     done
done