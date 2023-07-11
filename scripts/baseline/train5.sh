#!/usr/bin/env bash

DATASET=$1
SHOTS=$2

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 12800 1e-4 vit_b16_embedloss & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 12800 5e-5 vit_b16_embedloss & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 19200 5e-5 vit_b16_embedloss & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 25600 5e-5 vit_b16_embedloss & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 12800 1e-4 vit_b16_encoderloss & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 12800 5e-5 vit_b16_encoderloss & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 19200 5e-5 vit_b16_encoderloss & \

rlaunch --gpu=1 --cpu=8 --memory=16000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/onegpu.sh ${DATASET} ${SHOTS} 25600 5e-5 vit_b16_encoderloss & \