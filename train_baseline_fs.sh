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