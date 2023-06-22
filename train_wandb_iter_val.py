import argparse
import torch
import torch.distributed as dist
import os

from datasets import DataManager, TensorDataset
from torch.utils.data import DataLoader
from processor import train_wandb, train_lpclip, \
    train_wandb_two_stage, train_wandb_iter, \
    train_wandb_iter_wiseft, train_wandb_iter_wiseft_val, \
    train_caption, train_wandb_iter_wiseft_val_fixedfirst
from tools.utils import set_random_seed, collect_env_info
from tools.logger import setup_logger
from tools.train_utils import *

import wandb
import warnings
warnings.filterwarnings("ignore")


dataset_name = {
    "OxfordPets": "oxford_pets",
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "StanfordCars": "stanford_cars",
    "Food101": "food101",
    "SUN397": "sun397",
    "Caltech101": "caltech101",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",
    "ImageNetV2": "imagenetv2",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r"
}


def main(args):
    cfg = setup_cfg(args)
    logger = setup_logger(cfg.TRAINER.NAME, cfg.OUTPUT_DIR, if_train=True)

    run = wandb.init(project=args.wandb_proj)    # 'baseline_caption' baseline_ablation  baseline_cattn_vocabloss
    # run.name = 'vitb16-' + cfg.DATASET.NAME + f'-{cfg.DATASET.NUM_SHOTS}s-{cfg.TRAINER.NAME}-{cfg.OPTIM.NAME}-lr{cfg.OPTIM.LR}-e{cfg.OPTIM.MAX_EPOCH}'
    # run.name = 'vitb16-' + cfg.DATASET.NAME + f'-{cfg.DATASET.NUM_SHOTS}s-{cfg.TRAINER.NAME}-dp{cfg.MODEL.BONDER.DEPTH}-q{cfg.MODEL.BONDER.NUM_Q}' \
    #     f'-{cfg.OPTIM.NAME}-bs{cfg.DATALOADER.TRAIN_X.BATCH_SIZE}' \
    #     f'-lr{cfg.OPTIM.LR}-it{cfg.OPTIM.MAX_ITER}-warmit{cfg.OPTIM.WARMUP_ITER}'

    run.name = f'tmlp_lora-{cfg.MODEL.BACKBONE.NAME}-{cfg.DATASET.NAME}-{cfg.DATASET.NUM_SHOTS}s-{cfg.TRAINER.NAME}-r{cfg.MODEL.LORA.RANK}' \
        f'-a{cfg.MODEL.LORA.ALPHA}-{cfg.MODEL.TEXT.ENCODER}-{cfg.INPUT.TEXT_AUG}' \
        f'-iter{cfg.OPTIM.MAX_ITER}-lr{cfg.OPTIM.LR}-bs{cfg.DATALOADER.TRAIN_X.BATCH_SIZE}' \
        f'-dp{cfg.MODEL.BONDER.DEPTH}-q{cfg.MODEL.BONDER.NUM_Q}' \
        f'-{cfg.OPTIM.NAME}-warmit{cfg.OPTIM.WARMUP_ITER}-seed{cfg.SEED}'
    # run.name = 'all_mlp_lora'

    if cfg.SEED >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    if cfg.TRAIN.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
        pass
    else:
        print_args(args, cfg)
        logger.info("Collecting env info ...")
        logger.info("** System info **\n{}\n".format(collect_env_info()))

    # 1.dataset
    data = DataManager(cfg)

    # 2.model ( +optim +sche)
    model = MODELS[cfg.TRAINER.NAME](cfg, data.dataset.classnames)

    # prepare extracted features
    # features_root = "/data/run01/scz0bkt/datasets/recognition_features/image/ViT-B-16_0/"
    # features_file = f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.pth"
    # ccrop_features_path = os.path.join(features_root, cfg.DATASET.NAME, dataset_name[cfg.DATASET.NAME], 'none', features_file)
    # ccrop_features = torch.load(ccrop_features_path)
    #
    # image_features_path = os.path.join(features_root, cfg.DATASET.NAME, dataset_name[cfg.DATASET.NAME], 'flip_view_1', features_file)
    # image_features = torch.load(image_features_path)
    # train_features = torch.cat([ccrop_features['train']['features'], image_features['train']['features']], dim=0)
    # train_labels = torch.cat([ccrop_features['train']['labels'], image_features['train']['labels']], dim=0)
    #
    # image_train_dataset = TensorDataset(
    #     train_features,
    #     train_labels
    # )
    # image_val_dataset = TensorDataset(
    #     ccrop_features['val']['features'],
    #     ccrop_features['val']['labels']
    # )
    #
    # test_features_path = os.path.join(features_root, cfg.DATASET.NAME, dataset_name[cfg.DATASET.NAME], "test.pth")
    # test_features = torch.load(test_features_path)
    # test_dataset = TensorDataset(
    #     test_features['features'],
    #     test_features['labels']
    # )
    #
    # batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
    # image_loader = DataLoader(
    #     image_train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    #
    # val_loader = DataLoader(
    #     image_val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )
    #
    # test_batch_size = cfg.DATALOADER.TEST.BATCH_SIZE
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=test_batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    image_loader = data.train_loader
    val_loader = data.val_loader
    test_loader = data.test_loader

    # 3.train
    if cfg.TRAINER.NAME in ["lpclip", "lpsam"]:
        train_lpclip(cfg, model, data, args.local_rank)
    elif cfg.TRAINER.NAME in ["baseline_cattn_vocabloss_shembed_zsinit_fixedfirst"]:
        train_wandb_two_stage(cfg, model, data, args.local_rank)
    elif 'fixedfirst' in cfg.TRAINER.NAME:
        train_wandb_iter_wiseft_val_fixedfirst(cfg, model, data, image_loader, val_loader, test_loader, args.local_rank)
    elif "caption" in cfg.TRAINER.NAME:
        train_caption(cfg, model, data, image_loader, val_loader, test_loader, args.local_rank)
    elif ("wiseft" in cfg.TRAINER.NAME) or ("sattn" in cfg.TRAINER.NAME):
        train_wandb_iter_wiseft_val(cfg, model, data, image_loader, val_loader, test_loader, args.local_rank)
    else:
        train_wandb_iter(cfg, model, data, args.local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/sdb/tanhao/recognition/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="/mnt/sdb/tanhao/logs/Baseline/others", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--dist-train", type=bool, default=False, help="path to config file"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="baseline", help="name of trainer")  # CoOp
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--wandb-proj", type=str, default="baseline_caption", help="project name of wandb"
    )
    args = parser.parse_args()
    main(args)