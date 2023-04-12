import torch
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
import logging

from tools.utils import read_image
import torch.distributed as dist

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
    'SUN397': SUN397,
    'UCF101': UCF101
}

# output = {
#             "label": item['label'],
#             "impath": item['impath'],
#             "index": idx,
#             "img":
#         }

# imgs, label = zip(*batch)
# label = torch.tensor(label, dtype=torch.int64)
# return torch.stack(imgs, dim=0), label


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    # batch: [{}, {}, ...]
    images, labels = [], []
    for d in batch:
        images.append(d['img'])
        labels.append(torch.tensor(d['label'], dtype=torch.long))

    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


def test_collate_fn(batch):
    images, labels = [], []
    for d in batch:
        images.append(d['img'])
        labels.append(torch.tensor(d['label'], dtype=torch.long))

    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


class DataManager():
    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None):
        self.logger = logging.getLogger(cfg.TRAINER.NAME)
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            self.logger.info("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            self.logger.info("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # 1.dataset + transform
        dataset = FACTORY[cfg.DATASET.NAME](cfg)    # dataset.train,  dataset.val,  dataset.test
        train_set = DatasetWrapper(cfg, dataset.train, transform=tfm_train)
        test_set = DatasetWrapper(cfg, dataset.test, transform=tfm_test)

        # 2.dataloader
        test_batch = cfg.DATALOADER.TEST.BATCH_SIZE
        nw = cfg.DATALOADER.NUM_WORKERS
        if cfg.TRAIN.DIST_TRAIN:
            train_batch = cfg.DATALOADER.TRAIN_X.BATCH_SIZE // dist.get_world_size()
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=train_batch,
                                      pin_memory=True,
                                      num_workers=nw,
                                      shuffle=False,   # ddp, need to be False
                                      sampler=train_sampler,
                                      drop_last=False)
        else:
            train_batch = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
            train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=train_batch,
                                      sampler=train_sampler,
                                      num_workers=nw,
                                      drop_last=False)

        test_sampler = torch.utils.data.sampler.RandomSampler(test_set)
        test_loader = DataLoader(test_set,
                                 batch_size=test_batch,
                                 sampler=test_sampler,
                                 num_workers=nw,
                                 drop_last=False)

        # Attributes
        self._num_classes = dataset.num_classes
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset      # self.dataset.train, self.dataset.test
        self.train_loader = train_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME

        table = []
        table.append(["Dataset", dataset_name])

        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        self.logger.info(tabulate(table))


class DatasetWrapper(Dataset):
    def __init__(self, cfg, data_source, transform=None):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item['label'],
            "impath": item['impath'],
            "index": idx
        }

        img0 = read_image(item['impath'])

        if self.transform is not None:
            output["img"] = self.transform(img0)
        else:
            output["img"] = img0

        return output


if __name__ == '__main__':
    from train import setup_cfg
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/sdb/tanhao/recognition/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./output", help="output directory")
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
        "--config-file", type=str, default="../configs/trainers/Baseline/vit_b16_ep10_batch1.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="../configs/datasets/caltech101.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")  # CoOp
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

    cfg = setup_cfg(args)
    data = DataManager(cfg)
    for image, label in data.train_loader:
        print(image)
        print(label)
        break