for DATASET in caltech101 fgvc_aircraft dtd
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_t5/vit_b16_t5_base.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done