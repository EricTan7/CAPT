for DATASET in imagenet_wval
do
    for SHOTS in 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_bert/vit_b16_bert_base.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done