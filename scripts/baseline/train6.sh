for DATA in fgvc_aircraft
do
    for lr in 7e-5 2e-5
    do
        for iter in 25600 19200 12800
        do
            CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATA}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
            DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.4 MODEL.LORA.RANK 2 \
            SEED 1 DATA_SEED 1 \
            OPTIM.LR ${lr} OPTIM.MAX_ITER ${iter}
        done
    done
done