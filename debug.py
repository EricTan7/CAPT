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

import os
os.makedirs(os.path.dirname('./configs/test.txt'))
