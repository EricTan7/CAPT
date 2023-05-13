# import torch
# import torch.nn as nn

# from torch.optim.lr_scheduler import _LRScheduler
#
# AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]
#
#
# class _BaseWarmupScheduler(_LRScheduler):
#
#     def __init__(
#         self,
#         optimizer,
#         successor,
#         warmup_epoch,
#         last_epoch=-1,
#         verbose=False
#     ):
#         self.successor = successor
#         self.warmup_epoch = warmup_epoch
#         super().__init__(optimizer, last_epoch, verbose)
#
#     def get_lr(self):
#         raise NotImplementedError
#
#     def step(self, epoch=None):
#         if self.last_epoch >= self.warmup_epoch:
#             self.successor.step(epoch)
#             self._last_lr = self.successor.get_last_lr()
#         else:
#             super().step(epoch)
#
#
# class net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(12,3)
#     def forward(self, x):
#         return self.fc(x)
#
#
# class ConstantWarmupScheduler(_BaseWarmupScheduler):
#
#     def __init__(
#         self,
#         optimizer,
#         successor,
#         warmup_epoch,
#         cons_lr,
#         last_epoch=-1,
#         verbose=False
#     ):
#         self.cons_lr = cons_lr
#         super().__init__(
#             optimizer, successor, warmup_epoch, last_epoch, verbose
#         )
#
#     def get_lr(self):
#         if self.last_epoch >= self.warmup_epoch:
#             return self.successor.get_last_lr()
#         return [self.cons_lr for _ in self.base_lrs]
#
# model = net()
# optimizer_1 = torch.optim.Adam(model.parameters(), lr=0.02)
# scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=50)
# scheduler = ConstantWarmupScheduler(optimizer_1, scheduler_1, 1, 5e-4)
# print(f"初始的学习率：{optimizer_1.defaults['lr']}")
# for epoch in range(1,102):
#     for batch in range(50):
#         optimizer_1.zero_grad()
#         optimizer_1.step()
#     print(f"第 {epoch} epoch的lr:{optimizer_1.param_groups[0]['lr']}")
#     scheduler.step()

# import torch.nn.functional as F
# a = torch.rand((32,5,512))
# b = torch.rand((32,5,512))
# loss = F.cosine_similarity(a,b,dim=-1)
# print(loss.shape)
#
# lossnew = loss.mean()
# print(lossnew.shape)
# d = 1

# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# import numpy as np
# _tokenizer = _Tokenizer()
# a = _tokenizer.encode('i have a tench')
# b = _tokenizer.encode('goldfish')
# print(a)
# print(b)
# classnames = ['dog', 'cat', 'tench', 'photo']
# # tokens = [_tokenizer.encode(n) for n in classnames]
#
# tokens = []
# for n in classnames:
#     tokens.extend(_tokenizer.encode(n))
#
# map = dict(zip(tokens, list(range(len(tokens)))))
#
# t = _tokenizer.encode('photo dog')
# trs_t = [map[to] for to in t]
# d = 1
# print(map[1929])
# a = ['apple' + " X" * 32]
# print(a)

# a = torch.randn((2,3))
# b = a.view(-1)
# c = 1

# from clip.simple_tokenizer import MySimpleTokenizer
#
# classnames = ['dog', 'cat', 'tench', 'photo', 'a', 'of', '.']
# tokenizer = MySimpleTokenizer(classnames)
# a = tokenizer.forward('a photo of a tench.')
# b = 1

# from clip import clip
#
# clip_model, _ = clip.load("RN50", jit=False)
# clip_model.float()

# import os
# os.makedirs(os.path.dirname('./configs/test.txt'))


import argparse

from datasets import DataManager
from tools.train_utils import *


def main(args):
    cfg = setup_cfg(args)

    # 1.dataset
    data = DataManager(cfg)

    classnames = data.dataset.classnames
    with open(f'{cfg.DATASET.NAME}_classnames.txt', 'a') as f:
        for item in classnames:
            f.write(str(item) + '\n')


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
