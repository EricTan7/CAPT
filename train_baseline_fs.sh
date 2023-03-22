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



