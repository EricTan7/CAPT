CUDA_VISIBLE_DEVICES=7 python datasets/extraction/img_features.py \
--root /mnt/sdb/tanhao/recognition/ \
--seed 1 \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_mul.yaml \
DATASET.NUM_SHOTS 16
INPUT.IMG_VIEWS 1