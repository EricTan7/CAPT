for SHOTS in 16 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_add.yaml \
    DATASET.NUM_SHOTS ${SHOTS} TRAIN.TEST_FREQ 200 MODEL.BONDER.DEPTH 1
done
