# for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet sun397
rlaunch --gpu=8 --cpu=50 --memory=200000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train_8datasets.sh 1 & \

rlaunch --gpu=8 --cpu=50 --memory=200000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train_8datasets.sh 2 & \

rlaunch --gpu=8 --cpu=50 --memory=200000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train_8datasets.sh 4 & \

rlaunch --gpu=8 --cpu=50 --memory=200000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train_8datasets.sh 8 & \

rlaunch --gpu=8 --cpu=50 --memory=200000 --positive-tags=2080ti --max-wait-duration=24h -- bash /home/lijun07/code/CAPT/scripts/baseline/train_8datasets.sh 16 & \
