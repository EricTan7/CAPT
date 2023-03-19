


CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12315 train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/debug/ \
--dataset-config-file configs/datasets/caltech101.yaml \
--config-file configs/trainers/lpclip/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16 TRAIN.DIST_TRAIN True DATALOADER.TRAIN_X.BATCH_SIZE 64


CUDA_VISIBLE_DEVICES=7 python train.py \
--root /mnt/sdb/tanhao/recognition/ --seed 1 --output-dir /mnt/sdc/tanhao/prompt/lpclip/debug/ \
--dataset-config-file configs/datasets/caltech101.yaml \
--config-file configs/trainers/lpclip/rn50_ep50_batch32.yaml  \
DATASET.NUM_SHOTS 16