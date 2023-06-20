for DATA in fgvc_aircraft
do
    for alpha in 0.4 0.6 2.
    do
        for lr in 5e-5 2e-5 2e-4
        do
            CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATA}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
            DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA ${alpha} MODEL.LORA.RANK 2 \
            SEED 1 DATA_SEED 1 \
            OPTIM.LR ${lr}
        done
    done
done