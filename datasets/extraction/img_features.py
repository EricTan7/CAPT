import sys
sys.path.insert(0, '.')

import torch
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from configs import get_cfg_default
from datasets import DataManager
import argparse
from tools.utils import set_random_seed
import os
from tqdm import tqdm
from tools.train_utils import *

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


FACTORY = {
    'Caltech101': Caltech101,
    'DescribableTextures': DescribableTextures,
    'EuroSAT': EuroSAT,
    'FGVCAircraft': FGVCAircraft,
    'Food101': Food101,
    'ImageNet': ImageNet,
    'OxfordFlowers': OxfordFlowers,
    'OxfordPets': OxfordPets,
    'StanfordCars': StanfordCars,
    # 'SUN397': SUN397,
    'UCF101': UCF101
}

_tokenizer = _Tokenizer()
_MODELS = ["RN50", "RN101", "RN50x4", "ViT-B/32", "ViT-B/16"]


def makedirs(path):
    '''Make directories if not exist.'''
    if not os.path.exists(path):
        os.makedirs(path)


def get_img_features_path(cfg):
    root = cfg.DATASET.FEA_ROOT
    aug = "-".join(cfg.INPUT.TRANSFORMS)
    aug = aug + f'-view_{cfg.INPUT.IMG_VIEWS}'
    file = f'shots_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.pth'
    spath = os.path.join(root, 'image', cfg.MODEL.BACKBONE.NAME.replace('/', '-'), cfg.DATASET.NAME, aug, file)
    return spath


def get_test_img_features_path(cfg):
    root = cfg.DATASET.FEA_ROOT
    spath = os.path.join(root, 'image', cfg.MODEL.BACKBONE.NAME.replace('/', '-'), cfg.DATASET.NAME, 'test.pth')
    return spath


def extract_features(image_encoder, loader, num_views=1):
    features_dict = {
        'features': torch.Tensor(),
        'labels': torch.Tensor(),
        'paths': [],
    }

    ########################################
    # Start Feature Extractor
    ########################################
    image_encoder.eval()

    with torch.no_grad():
        for _ in range(num_views):
            for batch_idx, batch in enumerate(tqdm(loader)):
                data = batch["img"].cuda()
                all_x, cls = image_encoder(data)  # This is not L2 normed
                all_x = all_x.cpu()
                if batch_idx == 0:
                    features_dict['features'] = all_x
                    features_dict['labels'] = batch['label']
                    features_dict['paths'] = batch['impath']
                else:
                    features_dict['features'] = torch.cat((features_dict['features'], all_x), 0)
                    features_dict['labels'] = torch.cat((features_dict['labels'], batch['label']))
                    features_dict['paths'] = features_dict['paths'] + list(batch['impath'])
    return features_dict


def extract_img_features(cfg, clip_model, data, spath):
    makedirs(os.path.dirname(spath))

    if os.path.exists(spath):
        print(f"Features already saved at {spath}")
    else:
        print(f"Saving features to {spath}")

        print(f"Extracting features for train split ...")
        num_views = cfg.INPUT.IMG_VIEWS
        image_features = extract_features(clip_model.visual, data.train_loader, num_views)
        torch.save(image_features, spath)


def extract_test_img_features(cfg, clip_model, data, spath):
    makedirs(os.path.dirname(spath))

    if os.path.exists(spath):
        print(f"Features already saved at {spath}")
    else:
        print(f"Saving features to {spath}")

        print(f"Extracting features for test split ...")
        image_features = extract_features(clip_model.visual, data.test_loader, num_views=1)
        torch.save(image_features, spath)


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    print(f"Extracting features for {cfg.DATASET.NAME} {cfg.DATASET.NUM_SHOTS}shots with {cfg.INPUT.TRANSFORMS} aug and {cfg.INPUT.IMG_VIEWS}views.")
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # 1.dataset
    data = DataManager(cfg)

    # 2.model
    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, jit=False)
    clip_model.float()
    clip_model.eval()

    # 3.extract and save
    spath = get_img_features_path(cfg)
    extract_img_features(cfg, clip_model, data, spath)

    spath = get_test_img_features_path(cfg)
    extract_test_img_features(cfg, clip_model, data, spath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
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



