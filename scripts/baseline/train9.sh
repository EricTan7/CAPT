for ALPHA in 0.8 1. 2.
do
    for RANK in 16
    do
        for ITER in 19200 25600 76800
        do
            CUDA_VISIBLE_DEVICES=7 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
            DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA ${ALPHA} MODEL.LORA.RANK ${RANK} \
            SEED 1 DATA_SEED 1 OPTIM.MAX_ITER ${ITER}
        done
    done
done