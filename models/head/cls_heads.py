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

        if cfg.TRAINER.NAME.startswith('baseline'):
            self.fuse = cfg.TRAINER.BASELINE.FUSE
            assert self.fuse in ["cat", "mul"]
            if self.fuse == 'cat':
                in_fea = vis_dim * 2
                print(f"Using fuse method: cat")
            else:   # 'mul'
                in_fea = vis_dim
                print(f"Using fuse method: mul")
        else:
            in_fea = vis_dim

        self.fc = nn.Linear(in_fea, n_cls, bias=bias)
        # if cfg.TRAINER.BASELINE.PREC == "fp16":
        #     self.fc.half()

    def forward(self, image_fea, text_fea=None, logit_scale=None):   # [1,512] [1,512]
        image_fea = image_fea.float()
        # 1. fuse features
        if self.cfg.TRAINER.NAME.startswith('baseline'):
            if self.fuse == 'cat':
                fused_fea = torch.cat([text_fea, image_fea], dim=1)     # [1,1024]
            else:
                fused_fea = text_fea * image_fea    # [1,512]
                if logit_scale is not None:
                    fused_fea = logit_scale * fused_fea
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

        # self.transform = nn.Linear(vis_dim*2, vis_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(vis_dim*2, n_cls, bias=bias)

    def forward(self, image_fea, text_fea=None):   # [B,1024] [B,1024]
        image_fea = image_fea.float()
        # image_fea = self.sigmoid(self.transform(image_fea))
        fused_fea = torch.cat([image_fea, text_fea], dim=1)     # [B,2048]

        logits = self.fc(fused_fea)

        return logits


class ClsHead_fea_scale(nn.Module):
    def __init__(self, cfg, classnames, clip_model, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)
        self.cfg = cfg
        in_fea = vis_dim
        print(f"Using fuse method: mul")

        self.fc = nn.Linear(in_fea, n_cls, bias=bias)

    def forward(self, image_fea, text_fea, fea_scale=None):
        image_fea = image_fea.float()
        # 1. fuse features
        fused_fea = text_fea * image_fea
        if fea_scale is not None:
            fused_fea = fea_scale * fused_fea

        # 2. classification
        logits = self.fc(fused_fea)

        return logits


class ClsHead_logit_scale(nn.Module):
    def __init__(self, cfg, classnames, clip_model, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)
        self.cfg = cfg
        in_fea = vis_dim
        print(f"Using fuse method: mul")

        self.fc = nn.Linear(in_fea, n_cls, bias=bias)

    def forward(self, image_fea, text_fea, logit_scale):
        image_fea = image_fea.float()
        # 1. fuse features
        fused_fea = text_fea * image_fea

        # 2. classification
        logits = self.fc(fused_fea)
        logits = logit_scale * logits

        return logits