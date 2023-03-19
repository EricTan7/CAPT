# CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars rn50_ep50 end 16 1 False
# CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars rn50_ep100 end 16 2 False
# CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars rn50_ep100 end 16 4 False
# CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars rn50 end 16 8 False
# CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars rn50 end 16 16 False
# sed -i 's/\r//' train_coop_fs.sh

CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars vit_b16 end 16 16 False
CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars vit_b16_ep50 end 16 1 False
CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars vit_b16_ep100 end 16 2 False
CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars vit_b16_ep100 end 16 4 False
CUDA_VISIBLE_DEVICES=1 bash scripts/coop/main.sh stanford_cars vit_b16 end 16 8 False