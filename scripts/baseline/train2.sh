#!/usr/bin/env bash

DATASET=$1
SHOTS=$2

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 38400 2e-4 & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 38400 1e-4 & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 38400 5e-5 & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 38400 2e-5 & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 51200 2e-4 & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 51200 1e-4 & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 51200 5e-5 & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 51200 2e-5 & \