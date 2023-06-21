# fgvc_aircraft oxford_pets stanford_cars dtd eurosat oxford_flowers
for DATASET in eurosat
do
    for r in 2 4 8
    do
        for alpha in 0.4 0.6 1. 2.
        do
            CUDA_VISIBLE_DEVICES=5 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
            DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA ${alpha} MODEL.LORA.RANK ${r} \
            OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5
        done
    done
done