# for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet sun397
rlaunch --gpu=8 --cpu=32 --memory=128000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train1.sh fgvc_aircraft 12800 16 & \

rlaunch --gpu=8 --cpu=32 --memory=128000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train1.sh fgvc_aircraft 19200 16 & \

rlaunch --gpu=8 --cpu=32 --memory=128000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train1.sh stanford_cars 12800 16 & \

rlaunch --gpu=8 --cpu=32 --memory=128000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train1.sh stanford_cars 19200 16 & \

rlaunch --gpu=8 --cpu=32 --memory=128000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train1.sh eurosat 12800 16 & \

rlaunch --gpu=8 --cpu=32 --memory=128000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train1.sh eurosat 19200 16 & \