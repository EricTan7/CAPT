from .baseset import BaseNoriSet
import random, cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter
from .randaugment import rand_augment_transform
import nori2 as nori
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def aug_plus(dataset='ImageNet_LT', aug_type='randcls_sim', mode='train', randaug_n=2, randaug_m=10, plus_plus='False'):
    # PaCo's aug: https://github.com/jiequancui/ Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if plus_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_regular = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_sim02 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    if aug_type == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif aug_type == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif aug_type == 'sim_sim':
        transform_train = [transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randcls_sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim02':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim02)]

    if mode == 'train':
        return transform_train
    else:
        return val_transform

class ImageNet_LT_Nori(BaseNoriSet):
    def __init__(self, mode='train', json_path=None, transform=None):
        super(ImageNet_LT_Nori, self).__init__(mode, json_path, transform)
        self.class_dict = self._get_class_dict()
        self.transform = aug_plus(dataset='ImageNet_LT', aug_type='randcls_sim', mode=mode, plus_plus='False')

    def __getitem__(self, index):
        self._check_nori_fetcher()
        now_info = self.data[index]
        
        nori_id = now_info["nori_id"]
        img_bytes = self.nori_fetcher.get(nori_id)
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except:
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        if self.mode != 'train':
            image = self.transform(img)
        else:
            image = self.transform[0](img)
        
        image_label = now_info['category_id']  # 0-index
        return {
            'img': image,
            'label': image_label
        }

    def _check_nori_fetcher(self):
        """Lazy initialize nori fetcher. In this way, `NoriDataset` can be pickled and used
            in multiprocessing.
        """
        if self.nori_fetcher is None:
            self.nori_fetcher = nori.Fetcher()

class ImageNet_LT_data():
    def __init__(self, cfg) -> None:
        train_set = ImageNet_LT_Nori(mode='train',json_path='/home/lijun07/code/CAPT/datasets/LT/ImageNet_LT_train.json')
        test_set = ImageNet_LT_Nori(mode='valid',json_path='/home/lijun07/code/CAPT/datasets/LT/ImageNet_LT_val.json')

        if cfg.TRAIN.DIST_TRAIN:
            train_batch = cfg.DATALOADER.TRAIN_X.BATCH_SIZE // dist.get_world_size()
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=train_batch,
                                      pin_memory=True,
                                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                                      shuffle=False,   # ddp, need to be False
                                      sampler=train_sampler,
                                      drop_last=False)
        else:
            train_batch = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
            train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=train_batch,
                                      sampler=train_sampler,
                                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                                      drop_last=False)

        test_sampler = torch.utils.data.sampler.RandomSampler(test_set)
        test_loader = DataLoader(test_set,
                                 batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                                 sampler=test_sampler,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                 drop_last=False)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.classnames = []
        with open('/home/lijun07/code/CAPT/datasets/LT/classname.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                self.classnames.append(line.replace('_',' ').replace('(','').replace(')',''))