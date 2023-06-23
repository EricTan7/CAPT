#!/usr/bin/env bash

DATASET=$1
ITER=$2
SHOTS=$3

rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 0.4 2 & \
rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 0.6 2 & \
rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 1.0 2 & \
rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 2.0 2 & \
rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 0.4 4 & \
rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 0.6 4 & \
rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 1.0 4 & \
rlaunch --gpu=1 --cpu=16 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${ITER} ${SHOTS} 2.0 4 & \
