#!/bin/bash

# custom config
DATA=/mnt/sdb/tanhao/recognition/
TRAINER=Baseline

DATASET=$1
CFG=$2  # config file
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
FUSE=$5   # cat, mul
LR=$6     # 2e-3
WARM=$7     # 1e-5
BATCH=$8    #
EPOCH=$9

for SEED in 1 2 3
do
    # DIR=/mnt/sdc/tanhao/prompt/Baseline_v2/output/fs_learning/hyper_debug/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}/fuse_${FUSE}/seed${SEED}
    DIR=/mnt/sdc/tanhao/prompt/Baseline/output/fs_learning/hyper_debug/${DATASET}/nctx${NCTX}/fuse_${FUSE}/lr${LR}_warm${WARM}_batch${BATCH}_e${EPOCH}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.BASELINE.N_CTX ${NCTX} \
        TRAINER.BASELINE.FUSE ${FUSE} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done