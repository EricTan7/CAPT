import argparse
import torch
import torch.distributed as dist
import os
from datasets import DataManager
from processor import train_wandb, train_lpclip, train_wandb_two_stage
from tools.utils import set_random_seed, collect_env_info
from tools.logger import setup_logger
from tools.train_utils import *
from datasets.LT.ImageNet_LT_nori import ImageNet_LT_data
from nccl import configure_nccl

import wandb
import warnings
warnings.filterwarnings("ignore")


def main(args):
    cfg = setup_cfg(args)
    os.environ['LAUNCH_SITE']='hhd'
    configure_nccl()
    cfg.local_rank = int(os.environ["RANK"])
    if 'WORLD_SIZE' in os.environ:
        cfg.local_rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        print(f'WORLD_SIZE in environ: {cfg.world_size}')


    if cfg.TRAIN.DIST_TRAIN:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(cfg.local_rank)
    
    logger = setup_logger(cfg.TRAINER.NAME, cfg.OUTPUT_DIR, if_train=True)

    # run = wandb.init(project='baseline_cattn_vocabloss')
    if dist.get_rank() == 0:
        run = wandb.init(project='LT_baseline1')
        run.name = 'vitb16-' + cfg.DATASET.NAME + f'-{cfg.DATASET.NUM_SHOTS}s-{cfg.TRAINER.NAME}-{cfg.OPTIM.NAME}-lr{cfg.OPTIM.LR}-e{cfg.OPTIM.MAX_EPOCH}'
    # run = wandb.init(project='lpsam')
    # run.name = 'vitb16-' + cfg.DATASET.NAME + f'-{cfg.DATASET.NUM_SHOTS}s'
    # run.name = 'vitb16-' + cfg.DATASET.NAME + f'-{cfg.DATASET.NUM_SHOTS}s-{cfg.TRAINER.NAME}-{cfg.INPUT.NUM_VIEWS}v-{cfg.OPTIM.NAME}-lr{cfg.OPTIM.LR}-e{cfg.OPTIM.MAX_EPOCH}'

    if cfg.SEED >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True



    if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
        pass
    else:
        print_args(args, cfg)
        # logger.info("Collecting env info ...")
        # logger.info("** System info **\n{}\n".format(collect_env_info()))

    # 1.dataset
    data = ImageNet_LT_data(cfg)

    # 2.model ( +optim +sche)
    # try:
    #     model = MODELS[cfg.TRAINER.NAME](cfg, data.classnames)
    # except:
    #     raise TypeError(f"Trainer {cfg.TRAINER.NAME} is not available.")
    model = MODELS[cfg.TRAINER.NAME](cfg, data.classnames)

    # 3.train
    if cfg.TRAINER.NAME in ["lpclip", "lpsam"]:
        train_lpclip(cfg, model, data, cfg.local_rank)
    elif cfg.TRAINER.NAME in ["baseline_cattn_vocabloss_shembed_zsinit_fixedfirst"]:
        train_wandb_two_stage(cfg, model, data,cfg.local_rank)
    else:
        train_wandb(cfg, model, data, cfg.local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="/data/output/PT_LT/baseline1", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--dist-train", type=bool, default=True, help="path to config file"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
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
        "--config-file", type=str, default="/home/lijun07/code/CAPT/configs/trainers/Baseline_cattn_vocabloss/vit_b16_batch32_shembed_zsinit.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="/home/lijun07/code/CAPT/configs/datasets/imagenet.yaml",
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
    args = parser.parse_args()
    main(args)
