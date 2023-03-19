"""
Classification head
Separate, for the sake of multi-model unity
"""
import torch
import torch.nn as nn


class ClsHead_v1(nn.Module):
    def __init__(self, cfg, classnames, clip_model, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)
        self.cfg = cfg

        if cfg.TRAINER.NAME == 'baseline':
            self.fuse = cfg.TRAINER.BASELINE.FUSE
            assert self.fuse in ["cat", "mul"]
            if self.fuse == 'cat':
                in_fea = vis_dim * 2
            else:   # 'mul'
                in_fea = vis_dim
        else:
            in_fea = vis_dim

        self.fc = nn.Linear(in_fea, n_cls, bias=bias)
        # if cfg.TRAINER.BASELINE.PREC == "fp16":
        #     self.fc.half()

    def forward(self, image_fea, text_fea=None):   # [1,512] [1,512]
        image_fea = image_fea.float()
        # 1. fuse features
        if self.cfg.TRAINER.NAME == 'baseline':
            if self.fuse == 'cat':
                fused_fea = torch.cat([text_fea, image_fea], dim=1)     # [1,1024]
            else:
                fused_fea = text_fea * image_fea    # [1,512]
        else:
            fused_fea = image_fea

        # 2. classification
        logits = self.fc(fused_fea)

        return logits


class ClsHead_v2(nn.Module):
    def __init__(self, cfg, classnames, clip_model, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)
        self.cfg = cfg

        if cfg.TRAINER.NAME == 'baseline':
            self.fuse = cfg.TRAINER.BASELINE.FUSE
            assert self.fuse in ["cat", "mul"]
            if self.fuse == 'cat':
                in_fea = vis_dim * 2
            else:  # 'mul'
                in_fea = vis_dim
        else:
            in_fea = vis_dim

        self.fc = nn.Sequential(
            nn.Linear(in_fea, in_fea//2, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(in_fea//2, n_cls, bias=bias),
        )

    def forward(self, image_fea, text_fea=None):   # [1,512] [1,512]
        image_fea = image_fea.float()
        # 1. fuse features
        if self.cfg.TRAINER.NAME == 'baseline':
            if self.fuse == 'cat':
                fused_fea = torch.cat([text_fea, image_fea], dim=1)  # [1,1024]
            else:
                fused_fea = text_fea * image_fea  # [1,512]
        else:
            fused_fea = image_fea

        # 2. classification
        logits = self.fc(fused_fea)

        return logits