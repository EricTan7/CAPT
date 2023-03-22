


CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12315 train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/output/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/lpclip/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.DIST_TRAIN True DATALOADER.TRAIN_X.BATCH_SIZE 32


CUDA_VISIBLE_DEVICES=7 python train_sweep_hyper.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/sweep_hyper/ \
--dataset-config-file configs/datasets/caltech101.yaml \
--config-file configs/trainers/lpclip/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.PRINT_FREQ 20

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12315 train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/output/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/lpclip/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.DIST_TRAIN True DATALOADER.TRAIN_X.BATCH_SIZE 32


CUDA_VISIBLE_DEVICES=6 python train_wandb.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/otuput/ \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/lpclip/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.PRINT_FREQ 20 DATALOADER.TRAIN_X.BATCH_SIZE 32