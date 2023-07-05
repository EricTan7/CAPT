#!/usr/bin/env bash

DATASET=$1
SHOTS=$2
ITER=$3
LR=$4

gs=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} ${ITER} ${LR} & \rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-ta