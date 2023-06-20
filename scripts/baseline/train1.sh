for DATA in fgvc_aircraft oxford_pets
do
    for r in 1 2 4 8
    do
        for alpha in 0.4 1. 2. 8.
        do
            CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATA}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
            DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA ${alpha} MODEL.LORA.RANK ${r} \
            SEED 1 DATA_SEED 1
        done
    done
done