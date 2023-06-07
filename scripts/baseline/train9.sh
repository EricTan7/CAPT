CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_projector.yaml \
DATASET.NUM_SHOTS 16 TRAIN.TEST_FREQ 200

for DATASET in stanford_cars oxford_pets ucf101
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_projector.yaml \
        DATASET.NUM_SHOTS ${SHOTS} TRAIN.TEST_FREQ 100
    done
done