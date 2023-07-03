for DATASET in imagenet_wval
do
    for SHOTS in 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=7 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/rn50.yaml \
        DATASET.NUM_SHOTS ${SHOTS} OPTIM.MAX_ITER 12800 OPTIM.LR 1e-4
    done
done