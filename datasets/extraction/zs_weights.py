import sys
sys.path.insert(0, '.')

import torch
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from configs import get_cfg_default
from datasets import DataManager
import argparse
from tools.utils import set_random_seed
from datasets.templates import get_templates
import os
from tqdm import tqdm
from tools.train_utils import *
import torch.nn as nn
from torch.nn import functional as F

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


def get_zeroshot_features_path(cfg):
    root = cfg.DATASET.FEA_ROOT
    spath = os.path.join(root, 'zero_shot', cfg.DATASET.NAME, 'weights.pth')
    return spath


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def extract_zero_shot_weights(text_templates, classnames, in_features, clip_model, spath, device="cuda"):
    # imagenet_templates = [
    #     "itap of a {}.",
    #     "a bad photo of the {}.",
    #     "a origami {}.",
    #     "a photo of the large {}.",
    #     "a {} in a video game.",
    #     "art of the {}.",
    #     "a photo of the small {}."
    # ]

    # classname:["dog", "dsfa", ...]
    makedirs(os.path.dirname(spath))

    if os.path.exists(spath):
        print(f"Features already saved at {spath}")
    else:
        print(f"Saving features to {spath}")
        print(f"Extracting features for test split ...")
        num_classes = len(classnames)
        text_encoder = TextEncoder(clip_model)
        text_encoder.to(device)
        with torch.no_grad():
            # weights = torch.empty_like(cls_head.fc.weight.data)
            weights = torch.zeros(num_classes, in_features)
            for label in range(num_classes):
                text_prompts = [template.format(classnames[label]) for template in text_templates]
                text_tokenized = clip.tokenize(text_prompts)
                text_embedding = clip_model.token_embedding(text_tokenized).float()
                text_embedding = text_embedding.to(device)

                text_features = text_encoder(text_embedding, text_tokenized)
                text_features = text_features.mean(dim=0)  # average across all templates
                # text_features = torch.cat([text_features, text_features])
                weights[label] = text_features
            weights.data = F.normalize(weights, dim=1)
        torch.save(weights, spath)


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
    spath = get_zeroshot_features_path(cfg)
    text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
    in_features = clip_model.ln_final.weight.shape[0]
    extract_zero_shot_weights(text_templates, data.dataset.classnames, in_features, clip_model, spath)


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



