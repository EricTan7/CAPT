import argparse
import torch
import torch.distributed as dist
import os

from datasets import DataManager, TensorDataset
from torch.utils.data import DataLoader
from processor import train_wandb, train_lpclip, \
    train_wandb_two_stage, train_wandb_iter, \
    train_wandb_iter_wiseft, train_wandb_iter_wiseft_val, test_wiseft_val
from tools.utils import set_random_seed, collect_env_info
from tools.logger import setup_logger
from tools.train_utils import *

import warnings
warnings.filterwarnings("ignore")

device = 'cuda'

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
    logger = setup_logger(cfg.TRAINER.NAME, cfg.OUTPUT_DIR, if_train=False)

    if cfg.SEED >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    # 1.dataset
    data = DataManager(cfg)

    # 2.model
    model = MODELS[cfg.TRAINER.NAME](cfg, data.dataset.classnames)
    model.load_model(cfg.MODEL.INIT_WEIGHTS)    # load weights from imagenet-16s
    model.set_model_mode("test")
    model.to(device)

    # 3.test
    results, results_wiseft, _, test_loss, test_wiseft_loss, _ = test_wiseft_val(cfg, model, data.test_loader, ratio=0.5)
    test_acc, test_acc_wiseft, test_loss, test_loss_wiseft = results["accuracy"], results_wiseft["accuracy"], test_loss, test_wiseft_loss,
    info = []
    info += [f"test acc: {test_acc}"]
    info += [f"test acc (wiseft_0.5): {test_acc_wiseft}"]
    info += [f"test loss: {test_loss}"]
    info += [f"test loss (wiseft_0.5): {test_loss_wiseft}"]
    logger.info(" ".join(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/sdb/tanhao/recognition/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="/mnt/sdb/tanhao/logs/Baseline/", help="output directory")
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
    args = parser.parse_args()
    main(args)