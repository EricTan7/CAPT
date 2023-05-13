# sed -i 's/\r//' train_baseline_fs.sh
# DATASET=$1
# CFG=$2  # config file
# NCTX=$3  # number of context tokens
# SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
# FUSE=$5   # cat, mul
# LR=$6     # 2e-3
# WARM=$7     # 1e-5
# BATCH=$8
# EPOCH

# stanford_cars imagenet caltech101
CUDA_VISIBLE_DEVICES=1 bash scripts/baseline_v2/main.sh stanford_cars vit_b16_ep10_batch1 4 16 cat 2e-2 5e-4 1 10
CUDA_VISIBLE_DEVICES=0 bash scripts/baseline_v2/main.sh imagenet vit_b16_ep10_batch1 4 16 cat 2e-2 5e-4 1 50
CUDA_VISIBLE_DEVICES=7 bash scripts/baseline_v2/main.sh oxford_pets vit_b16_ep10_batch1 4 16 cat 2e-2 5e-4 1 50
CUDA_VISIBLE_DEVICES=6 bash scripts/baseline_v2/main.sh caltech101 vit_b16_ep10_batch1 8 16 cat 2e-2 5e-4 1 50
CUDA_VISIBLE_DEVICES=5 bash scripts/baseline_v2/main.sh oxford_flowers vit_b16_ep10_batch1 4 16 cat 2e-2 5e-4 1 50

# --root /mnt/sdb/tanhao/recognition/ --seed 1 --dataset-config-file configs/datasets/caltech101.yaml --config-file configs/trainers/Baseline/vit_b16_ep10_batch1.yaml --output-dir /mnt/sdc/tanhao/prompt/Baseline/debug/ DATASET.NUM_SHOTS 16 TRAINER.BASELINE.N_CTX 4 TRAINER.BASELINE.FUSE cat
CUDA_VISIBLE_DEVICES=6 bash scripts/baseline/main.sh caltech101 rn50_ep50 4 16 cat 2e-2 5e-4 1 50
CUDA_VISIBLE_DEVICES=6 bash scripts/baseline/main.sh caltech101 vit_b16_ep10_batch32 4 16 cat 1e-1 5e-3 32 50


CUDA_VISIBLE_DEVICES=3,4,5,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12315 train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/debug/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.N_CTX 4 TRAINER.BASELINE.FUSE cat TRAIN.DIST_TRAIN True DATALOADER.TRAIN_X.BATCH_SIZE 128

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12315 train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/debug/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.N_CTX 4 TRAINER.BASELINE.FUSE cat TRAIN.DIST_TRAIN True DATALOADER.TRAIN_X.BATCH_SIZE 64


CUDA_VISIBLE_DEVICES=7 python train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.N_CTX 4 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


CUDA_VISIBLE_DEVICES=7 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/sweep_hyper/ --dataset-config-file configs/datasets/caltech101.yaml --config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml DATASET.NUM_SHOTS 2 TRAINER.BASELINE.N_CTX 4 TRAINER.BASELINE.FUSE cat
TRAIN.PRINT_FREQ 20


CUDA_VISIBLE_DEVICES=3,4,5,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12315 train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/debug/ --dataset-config-file configs/datasets/imagenet.yaml --config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml DATASET.NUM_SHOTS 16 TRAINER.BASELINE.N_CTX 4 TRAINER.BASELINE.FUSE cat TRAIN.DIST_TRAIN True DATALOADER.TRAIN_X.BATCH_SIZE 128


CUDA_VISIBLE_DEVICES=7 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline/sweep_hyper/ \
--dataset-config-file configs/datasets/caltech101.yaml \
--config-file configs/trainers/Baseline/rn50_ep50_batch32.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.N_CTX 4 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=7 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_v2/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_v2/rn50_ep50_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=5 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_v2/debug/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_v2/rn50_ep50_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=5 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_v2/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_v2/rn50_ep50_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 10

CUDA_VISIBLE_DEVICES=7 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_v2/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_v2/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 10

CUDA_VISIBLE_DEVICES=5 python train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_v2/sweep_hyper/ \
--dataset-config-file configs/datasets/caltech101.yaml \
--config-file configs/trainers/Baseline_v2/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 10

CUDA_VISIBLE_DEVICES=5 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_v2/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_v2/rn50_batch32_aug.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 10


CUDA_VISIBLE_DEVICES=5 python train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn/debug/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=7 python train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/debug/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=5 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=7 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 8 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=2 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_mul.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE mul TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=3 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# fea scale
CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_mul.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE mul TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=7 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_mul.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE mul TRAIN.PRINT_FREQ 20


# random erasing
CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_aug.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE mul TRAIN.PRINT_FREQ 20


# zsinit
CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_fixed.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# cat + logit_scale
CUDA_VISIBLE_DEVICES=5 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_lscale.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# optimfc
CUDA_VISIBLE_DEVICES=5 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_optimfc.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20 OPTIM.LR_FC_RATIO 0.1


# add views
CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_views.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20 INPUT.NUM_VIEWS 2


# embed loss
CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_embedloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_embedloss/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# zsinit + 2x cattn
CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_2xcattn.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_2xcattn_pe.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# zsinit + cattn + post decoder
CUDA_VISIBLE_DEVICES=4 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vl_pd/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vl_pd/vit_b16_batch32.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# zsinit + cattn + two stage
CUDA_VISIBLE_DEVICES=5 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_fixedfirst.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# zsinit + cj,gb
CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_aug.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20


# zsinit + textaug
# 以后用imagenet + eurosat验证
CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_textaug.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/eurosat.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_textaug.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 3 \
OPTIM.LR 0.002 OPTIM.MAX_EPOCH 100


# zsinit + cvpr codebase hyperparams
CUDA_VISIBLE_DEVICES=7 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20 \
DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001

# zsinit + align to cvpr codebase
CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=4 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale.yaml \
DATASET.NUM_SHOTS 16 TRAINER.BASELINE.FUSE cat TRAIN.PRINT_FREQ 20 \
DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001


# # zsinit + align to cvpr codebase + iteration train
CUDA_VISIBLE_DEVICES=7 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001

CUDA_VISIBLE_DEVICES=6 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter.yaml \
DATASET.NUM_SHOTS 16


# previous 72.6 + wiseft
CUDA_VISIBLE_DEVICES=6 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001


# overfit when increase iters, so ablate the "warmup iter"
CUDA_VISIBLE_DEVICES=3 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 50

CUDA_VISIBLE_DEVICES=2 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 100

CUDA_VISIBLE_DEVICES=1 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 500

CUDA_VISIBLE_DEVICES=0 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 1000




# previous + mul
CUDA_VISIBLE_DEVICES=7 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft_mul.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 50

CUDA_VISIBLE_DEVICES=6 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft_mul.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 100

CUDA_VISIBLE_DEVICES=5 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft_mul.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 500

CUDA_VISIBLE_DEVICES=4 python train_wandb_iter.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/Baseline_cattn_vocabloss/sweep_hyper/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit_lscale_iter_wiseft_mul.yaml \
DATASET.NUM_SHOTS 16 DATALOADER.TRAIN_X.BATCH_SIZE 8 OPTIM.LR 0.0001 \
OPTIM.WARMUP_ITER 1000


# ablate: nxcattn
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=40afa4ca3f265a034bccdf4e176b2f2254081f21 WANDB_MODE=offline python train_wandb_iter.py \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/Baseline_cattn_vocabloss/vit_b16_shembed_zsinit_lscale_iter_wiseft_nxcattn.yaml \
DATASET.NUM_SHOTS 16 MODEL.BONDER.DEPTH 2


cd run/code/Baseline
sbatch scripts/baseline/train1.sh