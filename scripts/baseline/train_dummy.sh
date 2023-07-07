# for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet sun397

#!/bin/sh
#SBATCH --gpus=1

source activate ppt

# for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet sun397
for DATASET in fgvc_aircraft eurosat oxford_pets sun397
do
    for SHOTS in 16 8 4 2 1
    do
        WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python train_wandb_iter_val.py \
        --dataset-config-file /data/run01/scz0bkt/code/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /data/run01/scz0bkt/code/Baseline/configs/trainers/Baseline_sattn/vit_b16.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done



# 1. template mining / ensemble init
for DATASET in caltech101 oxford_pets imagenet
do
    for SHOTS in 16 8 4 2 1
    do
        python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done

for DATASET in caltech101 oxford_pets imagenet
do
    for SHOTS in 16 8 4 2 1
    do
        python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16.yaml \
        DATASET.NUM_SHOTS ${SHOTS} INPUT.TEXT_AUG ensemble
    done
done



# 2. add
for DATASET in caltech101 oxford_pets imagenet
do
    for SHOTS in 16 8 4 2 1
    do
        python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16_add.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done



# 3. template mining supervision  + 随机初始化
for DATASET in caltech101 oxford_pets imagenet
do
    for SHOTS in 16 8 4 2 1
    do
        python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16_ensemble.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done

for DATASET in imagenet_a imagenet_r imagenet_sketch imagenetv2
do
    python test_xdomain.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16.yaml \
    --output-dir /mnt/sdb/tanhao/logs/Baseline/xdomain/${DATASET}
done


python test_xdomain.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenetv2.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16.yaml \
--output-dir /mnt/sdb/tanhao/logs/Baseline/xdomain/${DATASET}


for SHOTS in 16 8 4 2 1
do
    python datasets/extraction/preprocess_caption.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/caltech101.yaml \
    DATASET.NUM_SHOTS ${SHOTS}
done


CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/caltech101.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16

for SHOTS in 16 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
    DATASET.NUM_SHOTS ${SHOTS}
done


CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/dtd.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 1



# 去掉category分支，caption单分支
for DATASET in caltech101 dtd eurosat fgvc_aircraft
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_caption_only.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done

# 双分支，caption分支去掉监督
for DATASET in caltech101 dtd eurosat fgvc_aircraft
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_wo_caption.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done

# multi stream实验，改用head做early stopping
for DATASET in caltech101 dtd eurosat fgvc_aircraft
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_headval.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done


CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16


# nxcattn for caption stream
CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_nx.yaml \
DATASET.NUM_SHOTS 16 MODEL.BONDER.DEPTH 6

CUDA_VISIBLE_DEVICES=5 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_nx.yaml \
DATASET.NUM_SHOTS 16 MODEL.BONDER.DEPTH 8



# imagenet_a imagenet_r
# for DATASET in imagenet_sketch imagenetv2
# do
#    python test_xdomain.py \
#    --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
#    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16.yaml \
#    --output-dir /mnt/sdb/tanhao/logs/Baseline/xdomain/${DATASET}
# done

for DATASET in dtd eurosat fgvc_aircraft
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done

for DATASET in dtd eurosat fgvc_aircraft
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done


# caption方法，跑剩下数据集的实验
# for DATASET in caltech101 dtd fgvc_aircraft food101 eurosat ucf101 oxford_flowers oxford_pets stanford_cars imagenet sun397
# sun397 cars pets ucf101
for DATASET in stanford_cars oxford_pets ucf101 sun397
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done

# food101 flowers
for DATASET in oxford_flowers food101
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done


# projector似乎有效
跑上剩下的数据集
stanford_cars oxford_pets ucf101
dtd eurosat fgvc_aircraft

# ============ 正在用train4.sh跑
for SHOTS in 8 4 2 1
do
    CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_projector.yaml \
    DATASET.NUM_SHOTS ${SHOTS} TRAIN.TEST_FREQ 200
done

for DATASET in caltech101 food101 oxford_flowers sun397
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_projector.yaml \
        DATASET.NUM_SHOTS ${SHOTS} TRAIN.TEST_FREQ 100
    done
done


# ==================== ablate 在细粒度数据集上是否真的带来提升 ============================
# fgvc_aircraft oxford_pets stanford_cars dtd
for DATASET in oxford_pets stanford_cars
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_wo_caption.yaml \
        DATASET.NUM_SHOTS ${SHOTS} TRAIN.TEST_FREQ 100
    done
done

for DATASET in oxford_pets stanford_cars
do
    for SHOTS in 16 8 4 2 1
    do
        CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_caption_only.yaml \
        DATASET.NUM_SHOTS ${SHOTS} TRAIN.TEST_FREQ 100
    done
done


# ==================== 不能完全肯定带来了提升，但是大部分是有提升 ===========================
# 下一步尝试在这些细粒度数据集上加深、逐层auxi + 调参
# fgvc_aircraft oxford_pets stanford_cars dtd eurosat oxford_flowers

for DATASET in oxford_pets stanford_cars
do
    for SHOTS in 16 8 4 2 1
    do
        for DEPTH in 2 4 6 8
        do
            CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_nx.yaml \
            DATASET.NUM_SHOTS ${SHOTS} MODEL.BONDER.DEPTH ${DEPTH}
        done
    done
done

for DATASET in oxford_pets stanford_cars
do
    for SHOTS in 16 8 4 2 1
    do
        for DEPTH in 2 4 6 8
            CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/${DATASET}.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream_nx_auxi.yaml \
            DATASET.NUM_SHOTS ${SHOTS} MODEL.BONDER.DEPTH ${DEPTH}
        done
    done
done


# ==================== debug t5 model =======================
CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/caltech101.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_t5/vit_b16_t5_base.yaml \
DATASET.NUM_SHOTS 16

# ==================== imagenet_wval ========================
CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16

# debug param
--dataset-config-file /home/tanhao/Baseline/configs/datasets/caltech101.yaml --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_t5/vit_b16_t5_base.yaml DATASET.NUM_SHOTS 16






# ============ debug lora ==========================
CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/caltech101.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16

for SHOTS in 16 1
do
    CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16
done

for SHOTS in 16 1
do
    CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/dtd.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS 16
done

# ============= hyper lora ======================
for SHOTS in 16 1
do
    for ALPHA in 0.2 0.4 0.6 0.8
    do
        for ITER in 12800 25600 38400
        do
            CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
            --dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
            --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
            DATASET.NUM_SHOTS 16 TRAIN.VAL_WISEFT False \
            OPTIM.MAX_ITER ${ITER} MODEL.LORA.ALPHA ${ALPHA}
        done
    done
done

# =============== seed ======================
for SHOTS in 16 1
do
    for SEED in 2 3 4 5 6 3407
    do
        CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/dtd.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
        DATASET.NUM_SHOTS 16 SEED ${SEED}
    done
done

CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16

CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16

CUDA_VISIBLE_DEVICES=4 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16

CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.


# ================= lora hyper ===========================
CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 16. MODEL.LORA.RANK 1

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.2 MODEL.LORA.RANK 8

CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.2 MODEL.LORA.RANK 16


# ================== 涨点配置 ===========================
CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.2 MODEL.LORA.RANK 16

# 换回0.1.1
CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

# lijun 配置
CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 8 MODEL.LORA.RANK 4

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 8 MODEL.LORA.RANK 4

CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 8 MODEL.LORA.RANK 4

# 继续调0.1.1，确认其他数据集是否涨点
# fgvc_aircraft oxford_pets stanford_cars dtd eurosat oxford_flowers
CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/oxford_pets.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/oxford_flowers.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

CUDA_VISIBLE_DEVICES=6 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/eurosat.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

# 确认0.1.1涨点
# 尝试正常的0.1.0是否涨点
CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/eurosat.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/oxford_pets.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1

# 0.1.0也有涨点
# 尝试0.1.1 + lijun重写类
CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 1. MODEL.LORA.RANK 1



# ======================= 调超参示例 ============================
for r in xxx
do
    for alpha in xxx
    do
        CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
        --dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
        --config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
        DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA ${alpha} MODEL.LORA.RANK ${r} \
    done
done

# onegpu.sh
DATASET=$1
SHOTS=$2
ALPHA=$3
RANK=$4

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/${DATASET}.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16.yaml \
    DATASET.NUM_SHOTS ${SHOTS} MODEL.LORA.ALPHA ${ALPHA} MODEL.LORA.RANK ${RANK} \
    OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5


# ====== fixed first: retain text encoder and fc but fixed first =======
# 可调：iter + fix_epoch, lr, shots
CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16_fixedfirst.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.4 MODEL.LORA.RANK 2 OPTIM.LR 5e-5

iter=19200
fix_iter=[2400, 4800, 6400]
lr=[1e-4, 5e-5]

# onegpu.sh
DATASET=$1
SHOTS=$2
FIX_ITER=$3
LR=$4

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/${DATASET}.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_lora/vit_b16_fixedfirst.yaml \
    DATASET.NUM_SHOTS ${SHOTS} MODEL.LORA.ALPHA 0.6 MODEL.LORA.RANK 4 \
    OPTIM.MAX_ITER 19200 TRAIN.FIX_EPOCH ${FIX_ITER} OPTIM.LR {LR}

# train2.sh
# meg_train.sh
全部改为train2.sh


# =========== params =================
CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/eurosat.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.6 MODEL.LORA.RANK 4

CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.6 MODEL.LORA.RANK 4

# =========== dtd r4_a0.6 ==============
CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/dtd.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 MODEL.LORA.ALPHA 0.6 MODEL.LORA.RANK 4


# =========== wo lora ================
CUDA_VISIBLE_DEVICES=3 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16


# ============ test seed and average =============
CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16 SIMPLE_SEED True OPTIM.MAX_ITER 600

CUDA_VISIBLE_DEVICES=6 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 600

CUDA_VISIBLE_DEVICES=6 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/stanford_cars.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 600


# ============ test only save lora params =============
CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/caltech101.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_lora/vit_b16.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 600

# =========== test se layer ================
CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_pre_all.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 600

CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_post.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 600



# se layer
# onegpu.sh

DATASET=$1
SHOTS=$2
ITER=$3
LR=$4

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=2a0ff77c64888b3bd539c7873069809fbfeb6059 WANDB_MODE=offline python /home/lijun07/code/CAPT/train_wandb_iter_val.py \
    --dataset-config-file /home/lijun07/code/CAPT/configs/datasets/${DATASET}.yaml \
    --config-file /home/lijun07/code/CAPT/configs/trainers/Baseline_caption/vit_b16_se_pre_all.yaml \
    DATASET.NUM_SHOTS ${SHOTS} OPTIM.MAX_ITER ${ITER} OPTIM.LR ${LR}


# =============== se_pre_all ===================
CUDA_VISIBLE_DEVICES=1 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_pre_all.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5

# =============== se_cross =====================
CUDA_VISIBLE_DEVICES=2 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_cross.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5

# =============== se_text ======================
CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_se_text.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5


# =============== rn50 =================
CUDA_VISIBLE_DEVICES=7 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/imagenet_wval.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/rn50.yaml \
DATASET.NUM_SHOTS 1 OPTIM.MAX_ITER 12800 OPTIM.LR 5e-5


# ============== abl: wo text supervision ============
CUDA_VISIBLE_DEVICES=0 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_textsup.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5

CUDA_VISIBLE_DEVICES=5 python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption_abl/vit_b16_wo_ctgsup.yaml \
DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 19200 OPTIM.LR 5e-5


# vit_large
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=40afa4ca3f265a034bccdf4e176b2f2254081f21 WANDB_MODE=offline python train_wandb_iter_val.py \
--dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
--config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_l14.yaml \
DATASET.NUM_SHOTS 16

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=40afa4ca3f265a034bccdf4e176b2f2254081f21 WANDB_MODE=offline python train_wandb_iter_val.py \
    --dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml \
    --config-file /home/tanhao/Baseline/configs/trainers/Baseline_caption/vit_b16_multi_stream.yaml \
    DATASET.NUM_SHOTS 16 OPTIM.MAX_ITER 400 OPTIM.LR 1e-4 SIMPLE_SEED True

for seed in 2 3
do
    for shots in 16 8 4 2 1
    do
        WANDB_API_KEY=40afa4ca3f265a034bccdf4e176b2f2254081f21 WANDB_MODE=offline python train_wandb_iter_val.py \
        --dataset-config-file /data/run01/scz0bkt/code/Baseline/configs/datasets/imagenet.yaml \
        --config-file /data/run01/scz0bkt/code/Baseline/configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32.yaml \
        DATASET.NUM_SHOTS ${SHOTS} SEED ${seed}
    done
done

