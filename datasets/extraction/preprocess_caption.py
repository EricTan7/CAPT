"""
Usage:
1. Generate captions for few-shot split:
    python datasets/extraction/preprocess_caption.py DATASET.NUM_SHOTS 16
2. Generate captions for whole training set, use default SHOTS=-1:
    python datasets/extraction/preprocess_caption.py --dataset-config-file /home/tanhao/Baseline/configs/datasets/fgvc_aircraft.yaml
"""
import sys
sys.path.insert(0, '.')
import logging
import argparse
import os
from tools.model import load_checkpoint

import torch
import torch.nn as nn
from torch.nn import functional as F

from solver import build_optimizer, build_scheduler, build_scheduler_iter
from models.head import *
from tools.train_utils import *
from tools.model import load_pretrained_weights

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import MySimpleTokenizer
from datasets.templates import get_templates
from datasets.transforms import build_transform
from datasets.caltech101 import Caltech101
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.food101 import Food101
from datasets.imagenet import ImageNet
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets
from datasets.stanford_cars import StanfordCars
from datasets.sun397 import SUN397
from datasets.ucf101 import UCF101
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenetv2 import ImageNetV2
from datasets.make_dataloader import FACTORY

_tokenizer = _Tokenizer()

dataset_name = {
    "OxfordPets": "oxford_pets",
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "StanfordCars": "stanford_cars",
    "Food101": "food-101",
    "SUN397": "sun397",
    "Caltech101": "caltech-101",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",
    "ImageNetV2": "imagenetv2",
    "ImageNetSketch": "imagenet-sketch",
    "ImageNetA": "imagenet-adversarial",
    "ImageNetR": "imagenet-rendition"
}


def main(args):
    cfg = setup_cfg(args)
    dataset = FACTORY[cfg.DATASET.NAME](cfg)  # dataset.train,  dataset.val,  dataset.test

    container = set()
    for item in dataset.train:
        impath = item['impath']
        imname = os.path.join(os.path.basename(os.path.dirname(impath)), os.path.basename(impath))
        container.add(imname)

    # caption
    # dict {"001.jpg": caption.}
    caption_path = os.path.join(cfg.DATASET.ROOT, dataset_name[cfg.DATASET.NAME], "captions_p2.txt")
    caption = dict()
    caption_tokenized = dict()
    with open(caption_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            if line[0] in container:
                caption[line[0]] = line[1]
                caption_tokenized[line[0]] = clip.tokenize(line[1])

    # extract captions for few-shot
    if cfg.DATASET.NUM_SHOTS > 0:
        processed_caption_path = os.path.join(cfg.DATASET.ROOT, dataset_name[cfg.DATASET.NAME], "split_fewshot_caption", f"captions_p2_train_{cfg.DATASET.NUM_SHOTS}s.txt")
        with open(processed_caption_path, 'w') as f:
            for key, item in caption.items():
                f.write(key+'\t'+item+'\n')
    # all captions for training set
    else:
        processed_caption_path = os.path.join(cfg.DATASET.ROOT, dataset_name[cfg.DATASET.NAME], f"captions_p2_train.txt")
        with open(processed_caption_path, 'w') as f:
            for key, item in caption.items():
                f.write(key + '\t' + item + '\n')

    # NOTE caltech training set, max length 55
    # sentence_len = [len(_tokenizer.encode(sen)) for key, sen in caption.items()]
    # print(max(sentence_len))


if __name__ == '__main__':
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
        "--config-file", type=str, default="/home/tanhao/Baseline/configs/trainers/Baseline_final_v1/vit_b16.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="/home/tanhao/Baseline/configs/datasets/caltech101.yaml",
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