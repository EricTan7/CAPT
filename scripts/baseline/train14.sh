for DATASET in caltech101 fgvc_aircraft dtd
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_bert/vit_b16_bert_large.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done