import logging
import torch
import torch.nn as nn
import numpy as np
from functools import partial

from solver import build_optimizer, build_scheduler
from .base import BaseModel
from models.head import ClsHead_lpsam

from .backbone import ImageEncoderViT, MODEL_PARAMS, ImageEncoderViT_tfout


class Linear_Probe(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cls_head = ClsHead_lpsam(classnames, 256)

        params = MODEL_PARAMS[cfg.MODEL.SAM.NAME]
        self.image_encoder = ImageEncoderViT(
            depth=params['encoder_depth'],
            embed_dim=params['encoder_embed_dim'],
            img_size=cfg.INPUT.SIZE[0],  # params['image_size'],
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=params['encoder_num_heads'],
            patch_size=params['vit_patch_size'],
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=params['encoder_global_attn_indexes'],
            window_size=14,
            out_chans=params['prompt_embed_dim'],
        )

        with open(cfg.MODEL.SAM.CHECKPOINT, "rb") as f:
            state_dict = torch.load(f)

        # equal interval sampling
        pos_embed_idx = np.linspace(0, 63, 14)
        pos_embed_idx = list(map(int, pos_embed_idx))
        rel_pos_idx = np.linspace(0, 126, 27)
        rel_pos_idx = list(map(int, rel_pos_idx))
        for k, v in state_dict.items():
            if k == 'pos_embed':
                v = v[:, pos_embed_idx, :, :]
                state_dict[k] = v[:, :, pos_embed_idx, :]
            elif ('rel_pos_w' in k) or ('rel_pos_h' in k):
                if v.shape[0] >=127:
                    state_dict[k] = v[rel_pos_idx, :]
        self.image_encoder.load_state_dict(state_dict)

        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))   # pooling to (1,1)

    def forward(self, image):  # image: [B,3,224,224]  label:[B]
        image_features = self.image_encoder(image)  # [B, 256, 14, 14]
        image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0, 2,1)  # [B, 196, 256]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # According to new repo: no L2 norm
        image_cls = image_features[:, 0, :]
        logits = self.cls_head(image_cls)  # [B,num_cls]

        return logits


class Linear_Probe_avgpool(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cls_head = ClsHead_lpsam(classnames, 256)

        params = MODEL_PARAMS[cfg.MODEL.SAM.NAME]
        self.image_encoder = ImageEncoderViT(
            depth=params['encoder_depth'],
            embed_dim=params['encoder_embed_dim'],
            img_size=cfg.INPUT.SIZE[0],  # params['image_size'],
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=params['encoder_num_heads'],
            patch_size=params['vit_patch_size'],
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=params['encoder_global_attn_indexes'],
            window_size=14,
            out_chans=params['prompt_embed_dim'],
        )

        with open(cfg.MODEL.SAM.CHECKPOINT, "rb") as f:
            state_dict = torch.load(f)

        # equal interval sampling
        pos_embed_idx = np.linspace(0, 63, 14)
        pos_embed_idx = list(map(int, pos_embed_idx))
        rel_pos_idx = np.linspace(0, 126, 27)
        rel_pos_idx = list(map(int, rel_pos_idx))
        for k, v in state_dict.items():
            if k == 'pos_embed':
                v = v[:, pos_embed_idx, :, :]
                state_dict[k] = v[:, :, pos_embed_idx, :]
            elif ('rel_pos_w' in k) or ('rel_pos_h' in k):
                if v.shape[0] >=127:
                    state_dict[k] = v[rel_pos_idx, :]
        self.image_encoder.load_state_dict(state_dict)

        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))   # pooling to (1,1)

    def forward(self, image):  # image: [B,3,224,224]  label:[B]
        image_features = self.image_encoder(image)  # [B, 256, 14, 14]
        image_features = self.final_pool(image_features)    # [B, 256, 1, 1]
        image_features = image_features.view(image_features.size(0), -1)
        # image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0, 2,1)  # [B, 196, 256]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # According to new repo: no L2 norm
        logits = self.cls_head(image_features)  # [B,num_cls]

        return logits


class Linear_Probe_tfout(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cls_head = ClsHead_lpsam(classnames, 768)

        params = MODEL_PARAMS[cfg.MODEL.SAM.NAME]
        self.image_encoder = ImageEncoderViT_tfout(
            depth=params['encoder_depth'],
            embed_dim=params['encoder_embed_dim'],
            img_size=cfg.INPUT.SIZE[0],  # params['image_size'],
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=params['encoder_num_heads'],
            patch_size=params['vit_patch_size'],
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=params['encoder_global_attn_indexes'],
            window_size=14,
            out_chans=params['prompt_embed_dim'],
        )

        with open(cfg.MODEL.SAM.CHECKPOINT, "rb") as f:
            state_dict = torch.load(f)

        # equal interval sampling
        pos_embed_idx = np.linspace(0, 63, 14)
        pos_embed_idx = list(map(int, pos_embed_idx))
        rel_pos_idx = np.linspace(0, 126, 27)
        rel_pos_idx = list(map(int, rel_pos_idx))
        for k, v in state_dict.items():
            if k == 'pos_embed':
                v = v[:, pos_embed_idx, :, :]
                state_dict[k] = v[:, :, pos_embed_idx, :]
            elif ('rel_pos_w' in k) or ('rel_pos_h' in k):
                if v.shape[0] >=127:
                    state_dict[k] = v[rel_pos_idx, :]
        self.image_encoder.load_state_dict(state_dict)

        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))   # pooling to (1,1)

    def forward(self, image):  # image: [B,3,224,224]  label:[B]
        image_features = self.image_encoder(image)  # [B, 14, 14, 768]
        image_features = self.final_pool(image_features.permute(0,3,1,2))    # [B, 768, 1, 1]
        image_features = image_features.view(image_features.size(0), -1)
        # image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0, 2,1)  # [B, 196, 256]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # According to new repo: no L2 norm
        logits = self.cls_head(image_features)  # [B,num_cls]

        return logits


class lpsam(BaseModel):
    def __init__(self, cfg, classnames=None):
        super().__init__()
        self.logger = logging.getLogger(cfg.TRAINER.NAME)   # lpclip
        self.check_cfg(cfg)
        self.cfg = cfg
        self.test_freq = cfg.TRAIN.TEST_FREQ

        self.logger.info("Building Linear Probe SAM")
        self.model = Linear_Probe_tfout(cfg, classnames)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["cls_head"]

        for name, param in self.model.named_parameters():
            if name_to_update[0] in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        self.logger.info(f"Parameters to be updated: {enabled}")

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.cls_head, cfg.OPTIM)
        self.sched = build_scheduler(self.optim, cfg.OPTIM)

        self.register_model("cls_head", self.model.cls_head, self.optim, self.sched)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image)     # logits
