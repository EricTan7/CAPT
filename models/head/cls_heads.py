"""
Classification head
Separate, for the sake of multi-model unity
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
from clip import clip
import torch.nn.functional as F
import numpy as np


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


class ClsHead_v2_lscale(nn.Module):
    def __init__(self, cfg, classnames, clip_model, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)
        self.cfg = cfg

        # self.transform = nn.Linear(vis_dim*2, vis_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(vis_dim*2, n_cls, bias=bias)

    def forward(self, image_fea, text_fea, logit_scale):   # [B,1024] [B,1024]
        image_fea = image_fea.float()
        # image_fea = self.sigmoid(self.transform(image_fea))
        fused_fea = torch.cat([image_fea, text_fea], dim=1)     # [B,2048]

        logits = self.fc(fused_fea)
        logits = logit_scale * logits

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


class ClsHead_sam(nn.Module):
    def __init__(self, classnames, vis_dim, ctx_dim, bias=False):
        super().__init__()
        n_cls = len(classnames)

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(vis_dim + ctx_dim, n_cls, bias=bias)

    def forward(self, image_fea, text_fea=None):   # [B,1024] [B,1024]
        image_fea = image_fea.float()
        text_fea = text_fea.float()
        fused_fea = torch.cat([image_fea, text_fea], dim=1)     # [B,2048]

        logits = self.fc(fused_fea)

        return logits


class ClsHead_lpsam(nn.Module):
    def __init__(self, classnames, vis_dim, bias=False):
        super().__init__()
        n_cls = len(classnames)

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(vis_dim, n_cls, bias=bias)

    def forward(self, image_fea):   # [B,1024] [B,1024]
        image_fea = image_fea.float()

        logits = self.fc(image_fea)

        return logits


class ClsHead_cat(nn.Module):
    def __init__(self, classnames, clip_model, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.fc = nn.Linear(vis_dim*2, n_cls, bias=bias)

    def forward(self, fused_fea):   # [B,1024] [B,1024]
        logits = self.fc(fused_fea)

        return logits


class ClsHead_cat_lscale(nn.Module):
    def __init__(self, classnames, clip_model, logit_scale, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.fc = nn.Linear(vis_dim*2, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, x):   # [B,1024] [B,1024]
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x


class ClsHead_mul_lscale(nn.Module):
    def __init__(self, classnames, clip_model, logit_scale, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.fc = nn.Linear(vis_dim, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, x):   # [B,1024] [B,1024]
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x


class ClsHead_add_lscale(nn.Module):
    def __init__(self, classnames, clip_model, logit_scale, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.fc = nn.Linear(vis_dim, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, x):   # [B,1024] [B,1024]
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x


class ClsHead_cat_lscale_lnable(nn.Module):
    def __init__(self, classnames, clip_model, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.fc = nn.Linear(vis_dim*2, n_cls, bias=bias)

        # Learnable
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # the same as CLIP
        self.logit_scale = nn.Parameter(torch.tensor(4.60517))

    def forward(self, x):   # [B,1024] [B,1024]
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x


class ClsHead_lscale(nn.Module):
    def __init__(self, classnames, clip_model, logit_scale, bias=False):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.fc = nn.Linear(vis_dim, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, x):   # [B,1024] [B,1024]
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x


class ClsHead_cat_lscale_img_text(nn.Module):
    def __init__(self, classnames, img_dim, text_dim, logit_scale, bias=False):
        super().__init__()
        n_cls = len(classnames)

        self.fc = nn.Linear(img_dim+text_dim, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, x):   # [B,1024] [B,1024]
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x


class ClsHead_cat_lscale_se_pre_all(nn.Module):
    def __init__(self, classnames, clip_model, logit_scale, bias=False, reduction=16):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.se_img = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // reduction, vis_dim, bias=False),
            nn.Sigmoid()
        )

        self.se_text = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // reduction, vis_dim, bias=False),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(vis_dim*2, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, img_fea, text_fea):   # [B,1024] [B,1024]
        img_fea = self.se_img(img_fea) * img_fea
        text_fea = self.se_text(text_fea) * text_fea

        x = torch.cat([img_fea, text_fea], dim=1)
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x


class ClsHead_cat_lscale_se_post(nn.Module):
    def __init__(self, classnames, clip_model, logit_scale, bias=False, reduction=16):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        self.se = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // reduction, vis_dim, bias=False),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(vis_dim * 2, n_cls, bias=bias)
        self.logit_scale = logit_scale

    def forward(self, x):  # [B,1024]
        x = self.se(x) * x
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x * self.logit_scale.exp()

        return x

class se_layer(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()

        self.se = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B,1024]
        return self.se(x) * x


class cross_se_layer(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()

        self.se_img = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
        self.se_text = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, text):  # [B,1024]
        # return self.se(x) * x
        img = self.se_img(text) * img
        text = self.se_text(img) * text
        return img, text


class cross_text_se_layer(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()

        self.se = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, text):  # [B,1024]
        text = self.se(img) * text
        return text