# workflow
# 1. read label names for each dataset

# 2. initialize clip

# 3. tokenize + embedding (label)

# 4. save: one file for each dataset

import torch
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from configs import get_cfg_default

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


if __name__ == '__main__':
    cfg = get_cfg_default()
    cfg.DATASET.ROOT = '/mnt/sdb/tanhao/recognition/'
    cfg.SEED = 1
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # clip_model, preprocess = clip.load("RN50", device=device)
    # dtype = clip_model.dtype
    # ctx_dim = clip_model.ln_final.weight.shape[0]

    for k, v in FACTORY.items():
        dataset = v(cfg)
        classnames = dataset.classnames
        classnames = [name.replace("_", " ").replace("(", "").replace(")", "") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # tokenized_label = clip.tokenize(classnames)
        # with torch.no_grad():
        #     label_embed = clip_model.token_embedding(tokenized_label).type(dtype)    # [n_cls, 77, 512]

        mask = []
        prompt_len = max(name_lens)
        print(f"{k} max len: {prompt_len}")
        # for i in range(len(name_lens)):
        #     if name_lens[i] > prompt_len:
        #         raise ValueError(f"Length of class {classnames[i]} {name_lens[i]} is larger than {prompt_len}.")
        #     else:
        #         mask_one = torch.ones(name_lens[i], ctx_dim)
        #         if name_lens[i] == prompt_len:
        #             mask.append(mask_one)
        #         else:
        #             mask_zero = torch.zeros(prompt_len - name_lens[i], ctx_dim)
        #             mask.append(torch.cat([mask_one, mask_zero], dim=0))
        # mask = torch.stack(mask).to(device)  # [n_cls, 10, 512]

        # label_info = {'embed': label_embed,
        #               'mask': mask}

        spath = f'{cfg.DATASET.NAME}.pt'



